import Foundation
import CoreML
import SwiftCoreMLTools
import UIKit
import Compression

public class NERF2DCoreML : ObservableObject {
    public enum BatchPreparationStatus {
        case notPrepared
        case preparing(count: Int)
        case ready
        
        var description: String {
            switch self {
            case .notPrepared:
                return "Not Prepared"
            case .preparing(let count):
                return "Preparing \(count)"
            case .ready:
                return "Ready"
            }
        }
    }
    
    public var trainingBatchProvider: MLBatchProvider?
    public var trainingBatchStatus = BatchPreparationStatus.notPrepared
    public var predictionBatchProvider: MLBatchProvider?
    public var predictionBatchStatus = BatchPreparationStatus.notPrepared
    public var modelStatus = "Train model"
    public var accuracy = "Accuracy: n/a"
    public var epoch: Int = 5

    var coreMLModelUrl: URL
    var coreMLCompiledModelUrl: URL?
    var model: MLModel?
    var retrainedModel: MLModel?
    
    private let batchSize: Int
    private let positionalEncodingLength: Int
    private let numberOfColors: Int = 4
    
    public init?(batchSize: Int, networkWidth: Int = 512, positionalEncodingLength: Int = 10) {
        self.coreMLModelUrl = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("NERF_Model")
            .appendingPathExtension("mlmodel")
        self.batchSize = batchSize
        self.positionalEncodingLength = positionalEncodingLength
        
        let coordsCount: Int = 2 * positionalEncodingLength * 2
    
        let coremlModel = Model(
            version: 4,
            shortDescription: "NERF-Trainable",
            author: "Gilevich Viacheslav",
            license: "MIT",
            userDefined: ["SwiftCoremltoolsVersion" : "master"]
        ) {
            Input(name: "coords", shape: [UInt(coordsCount)])
            Output(name: "colorOut", shape: [UInt(numberOfColors)], featureType: .float)
            TrainingInput(name: "coords", shape: [UInt(coordsCount)])
            TrainingInput(name: "color", shape: [UInt(numberOfColors)], featureType: .float)
            NeuralNetwork(
                losses: [MSE(name: "loss", input: "colorOut", target: "color")],
                optimizer: Adam(
                    learningRateDefault: 0.0001,
                    learningRateMax: 0.3,
                    miniBatchSizeDefault: UInt(batchSize),
                    miniBatchSizeRange: [UInt(batchSize)],
                    beta1Default: 0.9,
                    beta1Max: 1.0,
                    beta2Default: 0.999,
                    beta2Max: 1.0,
                    epsDefault: 0.00000001,
                    epsMax: 0.00000001
                ),
                epochDefault: UInt(self.epoch),
                epochSet: [UInt(self.epoch)],
                shuffle: false
            ) {
                InnerProduct(
                    name: "denseIn",
                    input: ["coords"],
                    output: ["denseInOutput"],
                    inputChannels: UInt(coordsCount),
                    outputChannels: UInt(networkWidth),
                    updatable: true
                )
                ReLu(
                    name: "reluIn",
                    input: ["denseInOutput"],
                    output: ["reluInOutput"]
                )
                
                InnerProduct(
                    name: "dense1",
                    input: ["reluInOutput"],
                    output: ["dense1Output"],
                    inputChannels: UInt(networkWidth),
                    outputChannels: UInt(networkWidth),
                    updatable: true
                )
                ReLu(
                    name: "relu1",
                    input: ["dense1Output"],
                    output: ["relu1Output"]
                )
                
                
                InnerProduct(
                    name: "dense2",
                    input: ["relu1Output"],
                    output: ["dense2Output"],
                    inputChannels: UInt(networkWidth),
                    outputChannels: UInt(networkWidth),
                    updatable: true
                )
                ReLu(
                    name: "relu2",
                    input: ["dense2Output"],
                    output: ["relu2Output"]
                )
                
                
                InnerProduct(
                    name: "dense3",
                    input: ["relu2Output"],
                    output: ["dense3Output"],
                    inputChannels: UInt(networkWidth),
                    outputChannels: UInt(networkWidth),
                    updatable: true
                )
                ReLu(
                    name: "relu3",
                    input: ["dense3Output"],
                    output: ["relu3Output"]
                )
                
                
                InnerProduct(
                    name: "denseOut",
                    input: ["relu3Output"],
                    output: ["colorOut"],
                    inputChannels: UInt(networkWidth),
                    outputChannels: UInt(numberOfColors),
                    updatable: true
                )
            }
        }
        
        guard let coreMLData: Data = coremlModel.coreMLData else { return nil }
        
        
        do {
            try coreMLData.write(to: coreMLModelUrl)
            let compiledModelURL = try MLModel.compileModel(at: coreMLModelUrl)
            
            self.coreMLCompiledModelUrl = compiledModelURL
            self.model = try MLModel(contentsOf: compiledModelURL)
        } catch {
            print(error)
            return nil
        }
        
    }
    
    func train(on image: UIImage, epochs: Int = 5, update: @escaping (UIImage?) -> Void) {
        guard let cgImage = image.cgImage else { return }
        
        guard let imageArray = image.fp32Array() else { return }
        
        var featureProviders = [MLFeatureProvider]()
        
        for y in 0..<cgImage.height {
            for x in 0..<cgImage.width {
                let xFract = Float32(x) / Float32(cgImage.width)
                let yFract = Float32(y) / Float32(cgImage.height)
                let coords = positionalEncodedInput(x: xFract, y: yFract)
                guard let coordsArray = try? MLMultiArray(shape: [coords.count as NSNumber], dataType: .float32),
                      let colorArray = try? MLMultiArray(shape: [numberOfColors as NSNumber], dataType: .float32)
                else { return }
                
                for i in 0..<coords.count {
                    coordsArray[i] = NSNumber(value: coords[i])
                }
                
                for i in 0..<numberOfColors {
                    colorArray[i] = imageArray[cgImage.width * y * numberOfColors + numberOfColors * x + i] as NSNumber
                }
                
                let inputValue = MLFeatureValue(multiArray: coordsArray)
                let outputValue = MLFeatureValue(multiArray: colorArray)
                
                let dataPointFeatures: [String: MLFeatureValue] = ["coords": inputValue, "color": outputValue]
                if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                    featureProviders.append(provider)
                }
            }
        }
        
        let batchProvider = MLArrayBatchProvider(array: featureProviders)
        
        print("Batch Providers done")
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        configuration.parameters?[.epochs] = epochs
        
        let progressHandler = { [weak self] (context: MLUpdateContext) in
            switch context.event {
            case .trainingBegin:
                print("Training started..")
            case .miniBatchEnd:
                
                if let batchIndex = context.metrics[.miniBatchIndex] as? Int,
                   let batchLoss = context.metrics[.lossValue] as? Double
                {
                    print("Mini batch \(batchIndex), loss: \(batchLoss)")
                }
            case .epochEnd:
                self?.image(
                    from: context.model,
                    batchProvider: batchProvider,
                    width: cgImage.width,
                    height: cgImage.height,
                    completion: update
                )

            default:
                print("Unknown event")
            }
        }

        let completionHandler = { [weak self] (context: MLUpdateContext) in
            print("Training completed with state \(context.task.state.rawValue)")
            print("CoreML Error: \(context.task.error.debugDescription)")
            
            if context.task.state != .completed {
                print("Failed")
                
                return
            }

            if let trainLoss = context.metrics[.lossValue] as? Double {
                print("Final loss: \(trainLoss)")
            }

            self?.retrainedModel = context.model
            
            self?.image(
                from: context.model,
                batchProvider: batchProvider,
                width: cgImage.width,
                height: cgImage.height,
                completion: update
            )
            

            print("Model Trained!")
        }

        let handlers = MLUpdateProgressHandlers(
            forEvents: [.trainingBegin, .miniBatchEnd, .epochEnd],
            progressHandler: progressHandler,
            completionHandler: completionHandler
        )
        
        guard let url = coreMLCompiledModelUrl else { return }
        
        print("Update task init")
        
        guard let updateTask = try? MLUpdateTask(
            forModelAt: url,
            trainingData: batchProvider,
            configuration: configuration,
            progressHandlers: handlers
        ) else {
            print("Oops!")
            return
        }

        updateTask.resume()
    }
    
    private var positionalEncodedCache: [Int: [Float32]] = [:]
    private var cachedSize: (Int, Int) = (0,0)
    
    private func positionalEncodedInput(x: Float32, y: Float32) -> [Float32] {
        return (0..<positionalEncodingLength).flatMap { encodingPosition -> [Float32] in
            return [
                sinf(powf(2, Float(encodingPosition)) * .pi * Float(x)),
                cosf(powf(2, Float(encodingPosition)) * .pi * Float(x)),
                sinf(powf(2, Float(encodingPosition)) * .pi * Float(y)),
                cosf(powf(2, Float(encodingPosition)) * .pi * Float(y)),
            ]
        }
    }
    
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
            return self.positionalEncodedInput(x: coords.0, y: coords.1)
        }
        
        positionalEncodedCache[row] = input
        
        return input
    }
    
    private func image(
        from model: MLModel,
        batchProvider: MLArrayBatchProvider,
        width: Int,
        height: Int,
        completion: @escaping (UIImage?) -> Void
    ) {
        var resultArray = [Float32](repeating: 0, count: width * height * numberOfColors)
        
        let predictions = try! model.predictions(fromBatch: batchProvider)
        for index in 0..<predictions.count {
            let prediction = predictions.features(at: index)
            let color = prediction.featureValue(for: "colorOut")!.multiArrayValue!
            
            resultArray[index * 4 + 0] = Float32(color[0].floatValue)
            resultArray[index * 4 + 1] = Float32(color[1].floatValue)
            resultArray[index * 4 + 2] = Float32(color[2].floatValue)
            resultArray[index * 4 + 3] = Float32(color[3].floatValue)
        }
        DispatchQueue.main.async {
            completion(UIImage(width: width, height: height, fp32Array: resultArray))
        }
    }
    
}
