//
//  NERF2DMPSGraph.swift
//  metal-nerf-2d
//
//  Created by Vyacheslav Gilevich on 14.08.2022.
//

import UIKit
import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

func getRandomData(numValues: UInt, minimum: Float, maximum: Float) -> [Float] {
    return (1...numValues).map { _ in Float.random(in: minimum..<maximum) }
}

class NERF2DMPSGraph {
    
    var graph: MPSGraph
    var inputTensor: MPSGraphTensor
    var lossColorTensor: MPSGraphTensor
    var targetTrainingTensors: [MPSGraphTensor]
    var targetInferenceTensors: [MPSGraphTensor]
    var targetTrainingOps: [MPSGraphOperation]
    var targetInferenceOps: [MPSGraphOperation]
    
    private let batchSize: Int
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let positionalEncodingLength: Int
    private let numberOfColors = 4
    
    init?(batchSize: Int, networkWidth: Int = 256, networkDepth: Int = 8, positionalEncodingLength: Int = 10) {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard let commandQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = commandQueue
        self.batchSize = batchSize
        self.graph = MPSGraph()
        self.positionalEncodingLength = positionalEncodingLength
        
        let coordsCount: Int = 2 * positionalEncodingLength * 2
        
        self.inputTensor = graph.placeholder(shape: [batchSize as NSNumber, coordsCount as NSNumber], name: "input tensor")
        self.lossColorTensor = graph.placeholder(shape: [batchSize as NSNumber, numberOfColors as NSNumber], name: "loss color tensor")
        
        var variableTensors = [MPSGraphTensor]()
        
        let denseInTensor = NERF2DMPSGraph.addFullyConnectedLayer(
            graph: graph,
            sourceTensor: inputTensor,
            weightsShape: [coordsCount as NSNumber, networkWidth as NSNumber],
            hasActivation: true,
            variableTensors: &variableTensors
        )
        
        var lastTensor: MPSGraphTensor = denseInTensor
        
        for _ in 0..<networkDepth {
            lastTensor = NERF2DMPSGraph.addFullyConnectedLayer(
                graph: graph,
                sourceTensor: lastTensor,
                weightsShape: [networkWidth as NSNumber, networkWidth as NSNumber],
                hasActivation: true,
                variableTensors: &variableTensors
            )
        }
        
        let fc1Tensor = NERF2DMPSGraph.addFullyConnectedLayer(
            graph: graph,
            sourceTensor: lastTensor,
            weightsShape: [networkWidth as NSNumber, numberOfColors as NSNumber],
            hasActivation: false,
            variableTensors: &variableTensors
        )
        
        let lossTensor = graph.reductionSum(with: fc1Tensor, axis: -1, name: "loss tensor")
        
        self.targetInferenceTensors = [fc1Tensor]
        self.targetInferenceOps = []
        
        self.targetTrainingTensors = [lossTensor]
        self.targetTrainingOps = NERF2DMPSGraph.getAssignOperations(
            graph: graph,
            lossTensor: lossTensor,
            variableTensors: variableTensors
        )
        
    }
    
    func train(on image: UIImage, epochs: Int = 5, update: @escaping (UIImage?) -> Void) {
        guard let cgImage = image.cgImage else { return }
        
        assert(batchSize == cgImage.width)
        
        guard var imageArray = image.fp32Array() else { return }
        guard let outputPointer = imageArray.withUnsafeMutableBufferPointer({ $0.baseAddress }) else { return }
        
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
    
    private func runEpoch(cgImage: CGImage, imageArrayPointer: UnsafeMutablePointer<Float32>) -> Float32 {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return -1 }
        let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: commandBuffer)
        
        for batch in 0..<cgImage.height {
            
            var inputArray = positionalEncodedInput(width: cgImage.width, height: cgImage.height, row: batch)
            let inputDesc = MPSNDArrayDescriptor(
                dataType: .float32,
                shape: [batchSize as NSNumber, 2 /*x,y*/ * positionalEncodingLength * 2 /*sin,cos*/ as NSNumber]
            )

            let inputData = MPSNDArray(device: device, descriptor: inputDesc)
            inputData.label = "input data"
            inputData.writeBytes(&inputArray, strideBytes: nil)
            
            let outputDesc = MPSNDArrayDescriptor(
                dataType: .float32,
                shape: [batchSize as NSNumber, numberOfColors as NSNumber]
            )
            
            let outputData = MPSNDArray(device: device, descriptor: outputDesc)
            outputData.label = "output Data"
            
            let rowPointer = imageArrayPointer.advanced(by: cgImage.width * batch * 4)
            outputData.writeBytes(rowPointer, strideBytes: nil)
            
            let executionDesc = MPSGraphExecutionDescriptor()
            executionDesc.waitUntilCompleted = true
            executionDesc.completionHandler = { (resultsDictionary, nil) in
                var loss: Float32 = 0

                guard let tensor = self.targetTrainingTensors.first else { return }
                guard let lossTensorData = resultsDictionary[tensor] else { return }

                lossTensorData.mpsndarray().readBytes(&loss, strideBytes: nil)

                print(loss)
            }
            
            let feed = [inputTensor: MPSGraphTensorData(inputData), lossColorTensor: MPSGraphTensorData(outputData)]
            
            _ = graph.encode(
                to: mpsCommandBuffer,
                feeds: feed,
                targetTensors: targetTrainingTensors,
                targetOperations: targetTrainingOps,
                executionDescriptor: executionDesc
            )
        }
        
        mpsCommandBuffer.commit()
        mpsCommandBuffer.waitUntilCompleted()
        
        return -1
    }
    
    private func getImage(height: Int) -> UIImage? {
        var imageArray = Array<Float32>(repeating: 0, count: batchSize * height * 4)
        
        guard let imageArrayPointer = imageArray.withUnsafeMutableBytes({ $0.baseAddress }) else { return nil }
        
        for batch in 0..<height {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
            
            var inputArray = positionalEncodedInput(width: batchSize, height: height, row: batch)
            let inputDesc = MPSNDArrayDescriptor(
                dataType: .float32,
                shape: [batchSize as NSNumber, 2 /*x,y*/ * positionalEncodingLength * 2 /*sin,cos*/ as NSNumber]
            )

            let inputData = MPSNDArray(device: device, descriptor: inputDesc)
            inputData.writeBytes(&inputArray, strideBytes: nil)
            inputData.synchronize(on: commandBuffer)
            
            let outputDesc = MPSNDArrayDescriptor(dataType: .float32, shape: [batchSize as NSNumber, 4])
            let outputData = MPSNDArray(device: device, descriptor: outputDesc)
            let outputOffset = batchSize * batch * MemoryLayout<Float32>.size * 4
            
            
            let executionDesc = MPSGraphExecutionDescriptor()
            executionDesc.waitUntilCompleted = true

            executionDesc.completionHandler = { (resultsDictionary, nil) in
                guard let tensor = self.targetInferenceTensors.first else { return }
                guard let outputTensorData = resultsDictionary[tensor] else { return }
                
                let rowPointer = imageArrayPointer.advanced(by: outputOffset)

                outputTensorData.mpsndarray().readBytes(rowPointer, strideBytes: nil)
            }
            
            _ = graph.encode(
                to: MPSCommandBuffer(commandBuffer: commandBuffer),
                feeds: [inputTensor: MPSGraphTensorData(inputData), lossColorTensor: MPSGraphTensorData(outputData)],
                targetTensors: targetInferenceTensors,
                targetOperations: targetInferenceOps,
                executionDescriptor: executionDesc
            )
           
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

// MARK: Layer helpers
    
private extension NERF2DMPSGraph {
    
    static func addFullyConnectedLayer(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        weightsShape: [NSNumber],
        hasActivation: Bool,
        variableTensors: inout [MPSGraphTensor]
    ) -> MPSGraphTensor {
        assert(weightsShape.count == 2)

        var weightCount = 1
        for length in weightsShape {
            weightCount *= length.intValue
        }
        
        let biasCount = weightsShape[1].intValue
        
        let fc0WeightsValues = getRandomData(numValues: UInt(weightCount), minimum: -0.2, maximum: 0.2)
        let fc0BiasesValues = [Float](repeating: 0.1, count: biasCount)

        let fcWeights = graph.variable(
            with: Data(bytes: fc0WeightsValues, count: weightCount * 4),
            shape: weightsShape,
            dataType: .float32,
            name: "variable for fc weights"
        )
        let fcBiases = graph.variable(
            with: Data(bytes: fc0BiasesValues, count: biasCount * 4),
            shape: [biasCount as NSNumber],
            dataType: .float32,
            name: "variable for fc biases"
        )
        
        let fcTensor = graph.matrixMultiplication(primary: sourceTensor, secondary: fcWeights, name: "fc layer")
        
        let fcBiasTensor = graph.addition(fcTensor, fcBiases, name: "fc biases")
        
        variableTensors += [fcWeights, fcBiases]

        if !hasActivation {
            return fcBiasTensor
        }
        
        let fcActivationTensor = graph.reLU(with: fcBiasTensor, name: nil)

        return fcActivationTensor
    }
    
    static func getAssignOperations(
        graph: MPSGraph,
        lossTensor: MPSGraphTensor,
        variableTensors: [MPSGraphTensor]
    ) -> [MPSGraphOperation] {
        let gradTensors = graph.gradients(of: lossTensor, with: variableTensors, name: nil)
     
        var updateOps: [MPSGraphOperation] = []
        for (key, value) in gradTensors {
            for tensor in graph.adam(
                learningRate: graph.constant(0.0001, dataType: .float32),
                beta1: graph.constant(0.9, dataType: .float32),
                beta2: graph.constant(0.999, dataType: .float32),
                epsilon: graph.constant(1e-7, dataType: .float32),
                beta1Power: graph.constant(1, dataType: .float32),
                beta2Power: graph.constant(1, dataType: .float32),
                values: key,
                momentum: graph.constant(0.0, dataType: .float32),
                velocity: graph.constant(0.0, dataType: .float32),
                maximumVelocity: graph.constant(1.0, dataType: .float32),
                gradient: value,
                name: "adam optimizer"
            ) {
                let assign = graph.assign(key, tensor: tensor, name: "idk")
                
                updateOps += [assign]
            }
        }
        
        return updateOps
    }
    
}
