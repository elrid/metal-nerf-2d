//
//  NERF2DMPSNN.swift
//  metal-nerf-2d
//
//  Created by Vyacheslav Gilevich on 27.08.2022.
//

import Foundation
import MetalPerformanceShaders
import UIKit
import MetalKit

class NERF2DMPSNN {
    
    private struct NERF2DGraphWeights {
        let optimizer: MPSNNOptimizerAdam
        
        let denseInData: FCDataSource
        let denceIn: MPSCNNFullyConnected
        
        let denseHiddenData: [FCDataSource]
        let denseHidden: [MPSCNNFullyConnected]
        
        let denseOutData: FCDataSource
        let denseOut: MPSCNNFullyConnected
    }
    
    private struct NERF2DGraph {
        let denseInNode: MPSCNNFullyConnectedNode
        let denseInReluNode: MPSCNNNeuronReLUNode
        let denseHiddenNodes: [MPSCNNFullyConnectedNode]
        let denseHiddenReluNode: [MPSCNNNeuronReLUNode]
        let denseOutNode: MPSCNNFullyConnectedNode
        let lossNode: MPSCNNLossNode?
        
        func output() -> MPSNNFilterNode {
            return lossNode ?? denseOutNode
        }
    }
    
    
    private let batchSize: Int
    private let positionalEncodingLength: Int
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var nerfGraph: NERF2DGraphWeights
    
    private let numberOfColors = 4
    
    
    init?(batchSize: Int, networkWidth: Int = 256, networkDepth: Int = 8, positionalEncodingLength: Int = 10) {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard let commandQueue = device.makeCommandQueue() else { return nil }
        
        self.device = device
        self.commandQueue = commandQueue
        self.batchSize = batchSize
        self.positionalEncodingLength = positionalEncodingLength
        
        guard MPSSupportsMTLDevice(device) else { return nil }
        
        let coordsCount: Int = 2 * positionalEncodingLength * 2
        let numberOfColors: Int = 4
        
        let _learning_rate: Float = 0.0001;
        let _beta1: Double = 0.9
        let _beta2: Double = 0.999
        let _epsilon: Float = 1e-07
        let _t: Int = 1
        
        let desc = MPSNNOptimizerDescriptor(
            learningRate: _learning_rate,
            gradientRescale: 1.0,
            regularizationType: .None,
            regularizationScale: 1.0
        )
        
        let optimizer = MPSNNOptimizerAdam(
            device: device,
            beta1: _beta1,
            beta2: _beta2,
            epsilon: _epsilon,
            timeStep: _t,
            optimizerDescriptor: desc
        )
        
        let denseInData = FCDataSource(
            device: device,
            commandQueue: commandQueue,
            input: coordsCount,
            output: networkWidth,
            optimizer: optimizer
        )
        
        var denseHiddenData: [FCDataSource] = []
        for _ in 0..<networkDepth {
            denseHiddenData.append(FCDataSource(
                device: device,
                commandQueue: commandQueue,
                input: networkWidth,
                output: networkWidth,
                optimizer: optimizer
            ))
        }
        
        let denseOutData = FCDataSource(
            device: device,
            commandQueue: commandQueue,
            input: networkWidth,
            output: numberOfColors,
            optimizer: optimizer
        )
        
        
        let denseIn = MPSCNNFullyConnected(device: device, weights: denseInData)
        
        var denseHidden: [MPSCNNFullyConnected] = []
        for data in denseHiddenData {
            denseHidden.append(MPSCNNFullyConnected(device: device, weights: data))
        }
        
        let denseOut = MPSCNNFullyConnected(device: device, weights: denseOutData)
        
        self.nerfGraph = NERF2DGraphWeights(
            optimizer: optimizer,
            denseInData: denseInData,
            denceIn: denseIn,
            denseHiddenData: denseHiddenData,
            denseHidden: denseHidden,
            denseOutData: denseOutData,
            denseOut: denseOut
        )
        
    }
    
    func train(on image: UIImage, epochs: Int = 5, update: @escaping (UIImage?) -> Void) {
        guard let cgImage = image.cgImage else { return }
        
        
        guard var imageArray = image.fp32Array() else { return }
        
        assert(batchSize == cgImage.width)
        
        let graph = buildGraph(forTraining: true)
        let finalNode = graph.output()
        
        let lossExitPoints = finalNode.trainingGraph(withSourceGradient: nil) { gradientNode, _, _, _ in
            gradientNode.resultImage.format = .float32
        }
        
        guard let resultImage = lossExitPoints?.first?.resultImage else { return }
        guard let trainingGraph = MPSNNGraph(device: device, resultImage: resultImage, resultImageIsNeeded: true)
        else { return }
        
        trainingGraph.format = .float32
        
        for epoch in 0..<epochs {
            
            let startDate = Date()
            
            let loss = self.runEpoch(trainingGraph: trainingGraph, cgImage: cgImage, imageArray: &imageArray)
            
            if let image = self.getImage(height: cgImage.height) {
                update(image)
            }
            
            print("epoch", epoch, "completed in", Date().timeIntervalSince(startDate), "loss", loss)
        }
    }
    
    private func runEpoch(trainingGraph: MPSNNGraph, cgImage: CGImage, imageArray: inout [Float32]) -> Float32 {
        var aggregatedLoss = Float32(0)
        guard let outputPointer = imageArray.withUnsafeMutableBytes({ $0.baseAddress }) else { return -1 }
        
        for batch in 0..<cgImage.height {
            autoreleasepool {
                guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
                
                guard let batchImages = positionalEncodedInput(
                    width: cgImage.width,
                    height: cgImage.height,
                    row: batch
                ) else {
                    return
                }
                
                var batchStatesUnsafe: [MPSCNNLossLabels?] = [MPSCNNLossLabels?](repeating: nil, count: cgImage.width)
                
                DispatchQueue.concurrentPerform(iterations: cgImage.width) { x in
                    guard let labelsDescriptor = MPSCNNLossDataDescriptor(
                        data: Data(
                            bytesNoCopy: outputPointer.advanced(
                                by: batch * cgImage.bytesPerRow + x * numberOfColors * MemoryLayout<Float32>.size
                            ),
                            count: numberOfColors * MemoryLayout<Float32>.size,
                            deallocator: .none
                        ),
                        layout: MPSDataLayout.HeightxWidthxFeatureChannels,
                        size: MTLSizeMake(1, 1, 4)
                    ) else {
                        return
                    }
                    
                    let lossState = MPSCNNLossLabels(device: device, labelsDescriptor: labelsDescriptor)
                    batchStatesUnsafe[x] = lossState
                }
                
                let batchStates = batchStatesUnsafe.compactMap({ $0 })
                let lossImages = batchStates.map { $0.lossImage() }
                
                guard batchStates.count == batchStatesUnsafe.count else { return }
                
                guard let result = trainingGraph.encodeBatch(
                    to: commandBuffer,
                    sourceImages: [batchImages],
                    sourceStates: [batchStates]
                ) else {
                    return
                }
                
                MPSImageBatchSynchronize(result, commandBuffer)
                MPSImageBatchSynchronize(lossImages, commandBuffer)
                
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                
                var losses = [Float32](repeating: 0.0, count: result.count)
                guard let lossesPointer = losses.withUnsafeMutableBytes({ $0.baseAddress }) else { return }
                
                for (x, loss) in lossImages.enumerated() {
                    loss.texture.getBytes(
                        lossesPointer.advanced(by: x * MemoryLayout<Float32>.size),
                        bytesPerRow: MemoryLayout<Float32>.size,
                        from: MTLRegionMake2D(0, 0, 1, 1),
                        mipmapLevel: 0
                    )
                }
                
                aggregatedLoss += losses.reduce(0.0, { partialResult, loss -> Float32 in
                    return partialResult + abs(loss)
                })
            }
        }
            
        
        return aggregatedLoss
    }
    
    // MARK: Inference
    
    private func getImage(height: Int) -> UIImage? {
        var imageArray = Array<Float32>(repeating: 0, count: batchSize * height * 4)

        guard let imageArrayPointer = imageArray.withUnsafeMutableBytes({ $0.baseAddress }) else { return nil }
        
        autoreleasepool {
        
            let graph = buildGraph(forTraining: false)
            let finalNode = graph.output()
            
            guard let inferenceGraph = MPSNNGraph(
                device: device,
                resultImage: finalNode.resultImage,
                resultImageIsNeeded: true
            ) else { return }
            
            inferenceGraph.format = .float32
            
            for batch in 0..<height {
                guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
                
                guard let batchImages = positionalEncodedInput(width: batchSize, height: height, row: batch) else {
                    return
                }
                
                guard let result = inferenceGraph.encodeBatch(
                    to: commandBuffer,
                    sourceImages: [batchImages],
                    sourceStates: nil
                ) else {
                    return
                }
                
                MPSImageBatchSynchronize(result, commandBuffer)
                
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                
                for (x, image) in result.enumerated() {
                    image.texture.getBytes(
                        imageArrayPointer.advanced(
                            by: batchSize * 4 * MemoryLayout<Float32>.size * batch + x * MemoryLayout<Float32>.size * 4
                        ),
                        bytesPerRow: 4 * MemoryLayout<Float32>.size,
                        from: MTLRegionMake2D(0, 0, 1, 1),
                        mipmapLevel: 0
                    )
                }
            }
        }

        return UIImage(width: batchSize, height: height, fp32Array: imageArray)
    }

    // MARK: Positional encoding
    
    private var positionalEncodedCacheTextures: [Int: [MPSImage]] = [:]
    private var cachedSize: (Int, Int) = (0,0)
    private var isCacheEnabled: Bool = false // due to limit of number of textures, tested on Radeon VII, macOS 13.0
    
    private func positionalEncodedInput(width: Int, height: Int, row: Int) -> [MPSImage]? {
        if isCacheEnabled {
            if cachedSize.0 != width || cachedSize.1 != height {
                positionalEncodedCacheTextures = [:]
                cachedSize = (width, height)
            }
            
            if let cached = positionalEncodedCacheTextures[row] {
                return cached
            }
        }
        
        let xCoords = (0..<width).map { (Float32($0) / Float32(width)) }
        let yCoords = Array<Float32>(repeating: (Float32(row) / Float32(height)), count: width)
        
        let input = zip(xCoords, yCoords).flatMap { coords -> [Float32] in
            return (0..<positionalEncodingLength).flatMap { encodingPosition -> [Float32] in
                return [
                    sinf(powf(2, Float(encodingPosition)) * .pi * Float(coords.0)),
                    cosf(powf(2, Float(encodingPosition)) * .pi * Float(coords.0)),
                    sinf(powf(2, Float(encodingPosition)) * .pi * Float(coords.1)),
                    cosf(powf(2, Float(encodingPosition)) * .pi * Float(coords.1)),
                ]
            }
        }

        
        var batchImages: [MPSImage] = []
        var allTextures: [MTLTexture] = []
        
        guard let inputPointer = input.withUnsafeBytes({ $0.baseAddress }) else { return nil }
        
        for x in 0..<width {
            let textureDescriptor = MTLTextureDescriptor()
            textureDescriptor.textureType = .type2DArray
            textureDescriptor.width = 1
            textureDescriptor.height = 1
            textureDescriptor.arrayLength = positionalEncodingLength
            textureDescriptor.pixelFormat = .rgba32Float
            textureDescriptor.resourceOptions = [.cpuCacheModeWriteCombined]
            
            #if targetEnvironment(macCatalyst)
            textureDescriptor.resourceOptions.insert(.storageModeShared)
            #endif
            
            guard let inputTexture = device.makeTexture(descriptor: textureDescriptor) else { return nil }
            
            let offsetX = x * positionalEncodingLength * 4 * MemoryLayout<Float32>.size
            
            for i in 0..<(positionalEncodingLength) {
                inputTexture.replace(
                    region: MTLRegionMake2D(0, 0, 1, 1),
                    mipmapLevel: 0,
                    slice: i,
                    withBytes: inputPointer.advanced(by: offsetX + i * MemoryLayout<Float32>.size * 4),
                    bytesPerRow: MemoryLayout<Float32>.size * 4,
                    bytesPerImage: MemoryLayout<Float32>.size * 4
                )
            }
            allTextures.append(inputTexture)
            batchImages.append(MPSImage(texture: inputTexture, featureChannels: positionalEncodingLength * 4))
        }
        
        guard let buffer = commandQueue.makeCommandBuffer() else { return nil }
        
        guard let commandEncoder = buffer.makeBlitCommandEncoder() else { return nil }
        
        #if targetEnvironment(macCatalyst)
        for texture in allTextures {
            commandEncoder.synchronize(resource: texture)
        }
        #endif
        commandEncoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        
        if isCacheEnabled {
            positionalEncodedCacheTextures[row] = batchImages
        }
        
        return batchImages
    }
    
}

private extension NERF2DMPSNN {
    
    
    private func buildGraph(forTraining: Bool) -> NERF2DGraph {
        let denseInNode = MPSCNNFullyConnectedNode(source: MPSNNImageNode(handle: nil), weights: nerfGraph.denseInData)
        var lastImage = denseInNode.resultImage
        
        let denseInReluNode = MPSCNNNeuronReLUNode(source: lastImage, a: 0.01)
        lastImage = denseInReluNode.resultImage
        
        var denseHiddenNodes: [MPSCNNFullyConnectedNode] = []
        var denseReluNodes: [MPSCNNNeuronReLUNode] = []
        
        for denseHidden in nerfGraph.denseHiddenData {
            let denseHiddenNode = MPSCNNFullyConnectedNode(source: lastImage, weights: denseHidden)
            denseHiddenNodes.append(denseHiddenNode)
            lastImage = denseHiddenNode.resultImage
            let denseReluNode = MPSCNNNeuronReLUNode(source: lastImage, a: 0.01)
            denseReluNodes.append(denseReluNode)
            lastImage = denseReluNode.resultImage
        }
        
        let denseOut = MPSCNNFullyConnectedNode(source: lastImage, weights: nerfGraph.denseOutData)
        let loss: MPSCNNLossNode?
        
        if forTraining {
            let lossDesc = MPSCNNLossDescriptor(type: .meanAbsoluteError, reductionType: .sum)
            lossDesc.weight = 1.0 / Float(batchSize)
            loss = MPSCNNLossNode(source: denseOut.resultImage, lossDescriptor: lossDesc)
        } else {
            loss = nil
        }

        return NERF2DGraph(
            denseInNode: denseInNode,
            denseInReluNode: denseInReluNode,
            denseHiddenNodes: denseHiddenNodes,
            denseHiddenReluNode: denseReluNodes,
            denseOutNode: denseOut,
            lossNode: loss
        )
    }
    
}

class FCDataSource: NSObject, MPSCNNConvolutionDataSource {
    
    private let convDesc: MPSCNNConvolutionDescriptor
    private let sizeBiases: Int
    private let sizeWeights: Int
    
    private let vDescWeights: MPSVectorDescriptor
    private let weightMomentumVector: MPSVector
    private let weightVelocityVector: MPSVector
    private let weightVector: MPSVector
    private let vDescBiases: MPSVectorDescriptor
    private let biasMomentumVector: MPSVector
    private let biasVelocityVector: MPSVector
    private let biasVector: MPSVector
    
    private let convWtsAndBias: MPSCNNConvolutionWeightsAndBiasesState
    
    private let weightPointer: UnsafeMutableRawPointer
    private let weightMomentumPointer: UnsafeMutableRawPointer
    private let weightVelocityPointer: UnsafeMutableRawPointer
    private let biasPointer: UnsafeMutableRawPointer
    private let biasMomentumPointer: UnsafeMutableRawPointer
    private let biasVelocityPointer: UnsafeMutableRawPointer
    
    private let commandQueue: MTLCommandQueue
    private let optimizer: MPSNNOptimizerAdam
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, input: Int, output: Int, optimizer: MPSNNOptimizerAdam) {
        self.commandQueue = commandQueue
        self.optimizer = optimizer
        self.convDesc = MPSCNNConvolutionDescriptor(
            kernelWidth: 1,
            kernelHeight: 1,
            inputFeatureChannels: input,
            outputFeatureChannels: output
        )

        self.sizeBiases = output * MemoryLayout<Float32>.size
        let lenWeights = input * output
        self.sizeWeights = lenWeights * MemoryLayout<Float32>.size
        

        self.vDescWeights = MPSVectorDescriptor(length: lenWeights, dataType: .float32)
        self.weightMomentumVector = MPSVector(device: device, descriptor: vDescWeights)
        self.weightVelocityVector = MPSVector(device: device, descriptor: vDescWeights)
        self.weightVector = MPSVector(device: device, descriptor: vDescWeights)
        
        self.vDescBiases = MPSVectorDescriptor(length: output, dataType: .float32)
        self.biasMomentumVector = MPSVector(device: device, descriptor: vDescBiases)
        self.biasVelocityVector = MPSVector(device: device, descriptor: vDescBiases)
        self.biasVector = MPSVector(device: device, descriptor: vDescBiases)


        self.convWtsAndBias = MPSCNNConvolutionWeightsAndBiasesState(
            weights: weightVector.data,
            biases: biasVector.data
        )

        self.weightPointer = weightVector.data.contents()
        self.weightMomentumPointer = weightMomentumVector.data.contents()
        self.weightVelocityPointer = weightVelocityVector.data.contents()

        self.biasPointer = biasVector.data.contents()
        self.biasMomentumPointer = biasMomentumVector.data.contents()
        self.biasVelocityPointer = biasVelocityVector.data.contents()
        
        for i in 0..<lenWeights {
            let ptr = self.weightPointer.assumingMemoryBound(to: Float32.self).advanced(by: i)
            ptr.pointee = 0.125 - Float32(arc4random_uniform(1024 * 1024)) / 1024.0 / 1024.0 / 4.0
        }
        
        for i in 0..<output {
            let ptr = self.biasPointer.assumingMemoryBound(to: Float32.self).advanced(by: i)
            ptr.pointee = 0.125 - Float32(arc4random_uniform(1024 * 1024)) / 1024.0 / 1024.0 / 4.0
        }
        
        #if targetEnvironment(macCatalyst)
        self.weightVector.data.didModifyRange(0..<self.sizeWeights)
        self.biasVector.data.didModifyRange(0..<self.sizeBiases)
        #endif
    }
    
    func dataType() -> MPSDataType {
        .float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        convDesc
    }
    
    func weights() -> UnsafeMutableRawPointer {
        weightPointer
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        biasPointer.assumingMemoryBound(to: Float.self)
    }
    
    func load() -> Bool {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return false }
        convWtsAndBias.synchronize(on: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return true
    }
    
    func purge() {
    }
    
    func label() -> String? {
        return "FC: \(convDesc.inputFeatureChannels) x \(convDesc.outputFeatureChannels)"
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        return self
    }
    
    func update(
        with commandBuffer: MTLCommandBuffer,
        gradientState: MPSCNNConvolutionGradientState,
        sourceState: MPSCNNConvolutionWeightsAndBiasesState
    ) -> MPSCNNConvolutionWeightsAndBiasesState? {
        optimizer.encode(
            commandBuffer: commandBuffer,
            convolutionGradientState: gradientState,
            convolutionSourceState: sourceState,
            inputMomentumVectors: [weightMomentumVector, biasMomentumVector],
            inputVelocityVectors: [weightVelocityVector, biasVelocityVector],
            resultState: convWtsAndBias
        )
        
        return convWtsAndBias;
    }
    
    
}
