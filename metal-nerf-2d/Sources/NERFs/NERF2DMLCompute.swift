//
//  NERF2DMLCompute.swift
//  metal-nerf-2d
//
//  Created by Vyacheslav Gilevich on 14.08.2022.
//

import Foundation
import MLCompute
import UIKit
import CoreGraphics

class NERF2DMLCompute {
    
    private struct NERF2DGraph {
        let inputTensor: MLCTensor
        
        let denseInWeightsTensor: MLCTensor
        let denseInBiasesTensor: MLCTensor
        
        let denseOutWeightsTensor: MLCTensor
        let denseOutBiasesTensor: MLCTensor
        
        let lossLabelTensor: MLCTensor
        let denseIn: MLCTensor
        let reluIn: MLCTensor
        let denseOut: MLCTensor
        
        let mlcGraph: MLCGraph
        let trainingGraph: MLCTrainingGraph
        let inferenceGraph: MLCInferenceGraph
        
    }
    
    private let batchSize: Int
    private let positionalEncodingLength: Int
    private let device: MLCDevice
    private var nerfGraph: NERF2DGraph
    
    init?(batchSize: Int, networkWidth: Int = 256, networkDepth: Int = 8, positionalEncodingLength: Int = 10) {
        guard let device = MLCDevice(type: .cpu) else { return nil }
        self.device = device
        self.batchSize = batchSize
        self.positionalEncodingLength = positionalEncodingLength
        
        let graph = MLCGraph()
        let coordsCount: Int = 2 * positionalEncodingLength * 2
        let numberOfColors: Int = 4
        
        guard let inputTensorDescriptor = MLCTensorDescriptor(shape: [batchSize, coordsCount, 1, 1], dataType: .float32),
              let denseInWeightsDescriptor = MLCTensorDescriptor(shape: [1, coordsCount * networkWidth, 1, 1], dataType: .float32),
              let denseInBiasesDescriptor = MLCTensorDescriptor(shape: [1, networkWidth, 1, 1], dataType: .float32),
              let denseMidWeightsDescriptor = MLCTensorDescriptor(shape: [1, networkWidth * networkWidth, 1, 1], dataType: .float32),
              let denseMidBiasesDescriptor = MLCTensorDescriptor(shape: [1, networkWidth, 1, 1], dataType: .float32),
              let denseOutWeightsDescriptor = MLCTensorDescriptor(shape: [1, networkWidth*numberOfColors, 1, 1], dataType: .float32),
              let denseOutBiasesDescriptor = MLCTensorDescriptor(shape: [1, numberOfColors, 1, 1], dataType: .float32),
              let lossLabelDescriptor =  MLCTensorDescriptor(shape: [batchSize, numberOfColors], dataType: .float32)
        else {
            return nil
        }
              
        let inputTensor = MLCTensor(descriptor: inputTensorDescriptor)
        let denseInWeightsTensor = MLCTensor(descriptor: denseInWeightsDescriptor, randomInitializerType: .glorotUniform)
        let denseInBiasesTensor = MLCTensor(descriptor: denseInBiasesDescriptor, randomInitializerType: .glorotUniform)
        
        let denseOutWeightsTensor = MLCTensor(descriptor: denseOutWeightsDescriptor, randomInitializerType: .glorotUniform)
        let denseOutBiasesTensor = MLCTensor(descriptor: denseOutBiasesDescriptor, randomInitializerType: .glorotUniform)

        let lossLabelTensor = MLCTensor(descriptor: lossLabelDescriptor)
   
        guard let denseInLayer = MLCFullyConnectedLayer(
            weights: denseInWeightsTensor,
            biases: denseInBiasesTensor,
            descriptor: MLCConvolutionDescriptor(
                kernelSizes: (height: coordsCount, width: networkWidth),
                inputFeatureChannelCount: coordsCount,
                outputFeatureChannelCount: networkWidth
            )
        ) else { return nil }

        guard let denseIn = graph.node(with: denseInLayer, sources: [inputTensor]) else { return nil }
        
        guard let activationDescriptor = MLCActivationDescriptor(type: MLCActivationType.relu) else { return nil }
        let activationLayer = MLCActivationLayer(descriptor: activationDescriptor)
        
        guard let reluIn = graph.node(with: activationLayer, source: denseIn) else { return nil }
        
        var lastTensor: MLCTensor = reluIn
        
        for _ in 0..<networkDepth {
            let denseMidWeightsTensor = MLCTensor(descriptor: denseMidWeightsDescriptor, randomInitializerType: .glorotUniform)
            let denseMidBiasesTensor = MLCTensor(descriptor: denseMidBiasesDescriptor, randomInitializerType: .glorotUniform)
            guard let denseMidLayer = MLCFullyConnectedLayer(
                weights: denseMidWeightsTensor,
                biases: denseMidBiasesTensor,
                descriptor: MLCConvolutionDescriptor(
                    kernelSizes: (height: networkWidth, width: networkWidth),
                    inputFeatureChannelCount: networkWidth,
                    outputFeatureChannelCount: networkWidth
                )
            ) else { return nil }
            guard let denseMid = graph.node(with: denseMidLayer, sources: [lastTensor]) else { return nil }
            
            guard let activationDescriptor = MLCActivationDescriptor(type: MLCActivationType.relu) else { return nil }
            let activationLayer = MLCActivationLayer(descriptor: activationDescriptor)
            
            guard let reluMid = graph.node(with: activationLayer, source: denseMid) else { return nil }
            
            lastTensor = reluMid
        }
        
        guard let denseOutLayer = MLCFullyConnectedLayer(
            weights: denseOutWeightsTensor,
            biases: denseOutBiasesTensor,
            descriptor: MLCConvolutionDescriptor(
                kernelSizes: (height: networkWidth, width: numberOfColors),
                inputFeatureChannelCount: networkWidth,
                outputFeatureChannelCount: numberOfColors
            )
        ) else { return nil }

        guard let denseOut = graph.node(with: denseOutLayer, sources: [lastTensor]) else { return nil }
        
        let trainingGraph = MLCTrainingGraph(
            graphObjects: [graph],
            lossLayer: MLCLossLayer(descriptor: MLCLossDescriptor(type: .meanAbsoluteError, reductionType: .sum)),
            optimizer: MLCAdamOptimizer(
                descriptor: MLCOptimizerDescriptor(
                    learningRate: 0.0001,
                    gradientRescale: 1.0,
                    regularizationType: .none,
                    regularizationScale: 0.0
                ),
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-7,
                timeStep: 1
            )
        )

        trainingGraph.addInputs(["coords" : inputTensor], lossLabels: ["pixel" : lossLabelTensor])
        trainingGraph.compile(options: [], device: device)
        
        
        let inferenceGraph = MLCInferenceGraph(graphObjects: [graph])
        inferenceGraph.addInputs(["coords" : inputTensor])
        inferenceGraph.compile(options: [], device: device)
        
        self.nerfGraph = NERF2DGraph(
            inputTensor: inputTensor,
            denseInWeightsTensor: denseInWeightsTensor,
            denseInBiasesTensor: denseInBiasesTensor,
            denseOutWeightsTensor: denseOutWeightsTensor,
            denseOutBiasesTensor: denseOutBiasesTensor,
            lossLabelTensor: lossLabelTensor,
            denseIn: denseIn,
            reluIn: reluIn,
            denseOut: denseOut,
            mlcGraph: graph,
            trainingGraph: trainingGraph,
            inferenceGraph: inferenceGraph
        )
    }
    
    func train(on image: UIImage, epochs: Int = 5, update: @escaping (UIImage?) -> Void) {
        guard let cgImage = image.cgImage else { return }
        
        assert(batchSize == cgImage.width)
        
        guard let imageArray = image.fp32Array() else { return }
        guard let outputPointer = imageArray.withUnsafeBufferPointer({ $0.baseAddress }) else { return }
        
        for epoch in 0..<epochs {
            
            let startDate = Date()
            
            let loss = self.runEpoch(cgImage: cgImage, imageArrayPointer: outputPointer)
            
            if epoch % 10 == 0 {
                if let image = self.getImage(height: cgImage.height) {
                    update(image)
                }
            }
            
            print("epoch", epoch, "completed in", Date().timeIntervalSince(startDate), "loss", loss)
        }
    }
    
    private func runEpoch(cgImage: CGImage, imageArrayPointer: UnsafePointer<Float32>) -> Float32 {
        var aggregatedLoss = Float32(0)
        
        for batch in 0..<cgImage.height {
            let inputArray = positionalEncodedInput(width: cgImage.width, height: cgImage.height, row: batch)
            
            guard let inputPointer = inputArray.withUnsafeBytes({ $0.baseAddress }) else { return -1 }
            let inputData = MLCTensorData(
                immutableBytesNoCopy: inputPointer,
                length: inputArray.count * MemoryLayout<Float32>.size
            )
            
            let rowPointer = imageArrayPointer.advanced(by: cgImage.width * batch * 4)
            let outputData = MLCTensorData(
                immutableBytesNoCopy: rowPointer,
                length: cgImage.width * MemoryLayout<Float32>.size * 4
            )
            
            nerfGraph.trainingGraph.execute(
                inputsData: ["coords": inputData],
                lossLabelsData: ["pixel": outputData],
                lossLabelWeightsData: nil,
                batchSize: batchSize,
                options: [.synchronous]
            ) { (r, e, time) in
                
                let unsafeLoss = r?.data?.withUnsafeBytes { ptr -> Float32? in
                    ptr.assumingMemoryBound(to: Float32.self).baseAddress?.pointee
                }
                
                if let loss = unsafeLoss {
                    aggregatedLoss += loss
                }
            }
        }
        
        return aggregatedLoss
    }
    
    private func getImage(height: Int) -> UIImage? {
        var imageArray = Array<Float32>(repeating: 0, count: batchSize * height * 4)
        
        guard let imageArrayPointer = imageArray.withUnsafeMutableBytes({ $0.baseAddress }) else { return nil }
        
        for batch in 0..<height {
            let inputArray = positionalEncodedInput(width: batchSize, height: height, row: batch)
            
            guard let inputPointer = inputArray.withUnsafeBytes({ $0.baseAddress }) else { return nil }
            let inputData = MLCTensorData(
                immutableBytesNoCopy: inputPointer,
                length: inputArray.count * MemoryLayout<Float32>.size
            )
            
            let rowPointer = imageArrayPointer.advanced(by: batchSize * batch * MemoryLayout<Float32>.size * 4)
            
            nerfGraph.inferenceGraph.execute(
                inputsData: ["coords" : inputData],
                batchSize: batchSize,
                options: [.synchronous]
            ) { [self] (r, e, time) in
                r?.copyDataFromDeviceMemory(
                    toBytes: rowPointer,
                    length: batchSize * MemoryLayout<Float32>.size * 4,
                    synchronizeWithDevice: device.actualDeviceType != .cpu
                )
            }
        }
        
        return UIImage(width: batchSize, height: height, fp32Array: imageArray)
    }
    
    private var positionalEncodedCache: [Int: [Float32]] = [:]
    private var cachedSize: (Int, Int) = (0,0)
    
    private func positionalEncodedInput(width: Int, height: Int, row: Int) -> [Float32] {
        if cachedSize.0 != width || cachedSize.1 != height {
            positionalEncodedCache = [:]
            cachedSize = (width, height)
        }
        
        if let cached = positionalEncodedCache[row] {
            return cached
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
        
        positionalEncodedCache[row] = input
        
        return input
    }
    
}
