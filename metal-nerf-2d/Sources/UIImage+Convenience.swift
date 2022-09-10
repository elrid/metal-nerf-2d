//
//  UIImage+Convenience.swift
//  metal-nerf-2d
//
//  Created by Vyacheslav Gilevich on 27.08.2022.
//

import UIKit
import Accelerate

extension UIImage {
    
    func uint8Array() -> [UInt8]? {
        guard let image = cgImage else { return nil }
        
        var array = Array<UInt8>(repeating: .zero, count: image.width * image.height * 4)
        
        guard let pointer = array.withUnsafeMutableBytes({ $0.baseAddress }) else { return nil }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB) else { return nil }
        
        guard let context = CGContext(
            data: pointer,
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: image.width * MemoryLayout<UInt8>.size * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
                | CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        
        context.translateBy(x: 0, y: CGFloat(image.height))
        context.scaleBy(x: 1, y: -1)
        
        UIGraphicsPushContext(context)
        draw(at: .zero)
        UIGraphicsPopContext()
        
        return array
    }
    
    convenience init?(width: Int, height: Int, uint8Array: [UInt8]) {
        let bufferPointer = uint8Array.withUnsafeBufferPointer({ $0 })
        let data = Data(buffer: bufferPointer)
        
        guard let dataProvider = CGDataProvider(data: data as CFData) else { return nil }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB) else { return nil }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * MemoryLayout<UInt8>.size * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(
                rawValue: CGBitmapInfo.byteOrder32Little.rawValue
                | CGImageAlphaInfo.premultipliedLast.rawValue
            ),
            provider: dataProvider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return nil
        }
        
        
        self.init(cgImage: cgImage)
    }
    
    func fp16Array() -> [UInt16]? {
        guard let image = cgImage else { return nil }
        
        var array = Array<UInt16>(repeating: .zero, count: image.width * image.height * 4)
        
        guard let pointer = array.withUnsafeMutableBytes({ $0.baseAddress }) else { return nil }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB) else { return nil }
        
        guard let context = CGContext(
            data: pointer,
            width: image.width,
            height: image.height,
            bitsPerComponent: 16,
            bytesPerRow: image.width * MemoryLayout<UInt16>.size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                | CGBitmapInfo.floatComponents.rawValue
                | CGImageByteOrderInfo.order16Little.rawValue
        ) else {
            return nil
        }
        
        context.translateBy(x: 0, y: CGFloat(image.height))
        context.scaleBy(x: 1, y: -1)
        
        UIGraphicsPushContext(context)
        draw(at: .zero)
        UIGraphicsPopContext()
        
        return array
    }
    
    convenience init?(width: Int, height: Int, fp16Array: [UInt16]) {
        let bufferPointer = fp16Array.withUnsafeBufferPointer({ $0 })
        let data = Data(buffer: bufferPointer)
        
        guard let dataProvider = CGDataProvider(data: data as CFData) else { return nil }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB) else { return nil }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 16,
            bitsPerPixel: 64,
            bytesPerRow: width * MemoryLayout<UInt16>.size * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(
                rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                    | CGBitmapInfo.floatComponents.rawValue
                    | CGImageByteOrderInfo.order16Little.rawValue
            ),
            provider: dataProvider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            return nil
        }
        
        
        self.init(cgImage: cgImage)
    }
    
    func fp32Array() -> [Float32]? {
        guard var baseArray = fp16Array() else { return nil }
        
        return baseArray.toFP32()
    }
    
    
    convenience init?(width: Int, height: Int, fp32Array: [Float32]) {
        let fp16Array = [UInt16](fp32Array: fp32Array)
        self.init(width: width, height: height, fp16Array: fp16Array)
    }
    
}


extension Array where Element == Float32 {
    
    mutating func toFP16() -> [UInt16] {
        var output = [UInt16](repeating: 0, count: self.count)
        let totalCount = count
        self.withUnsafeMutableBytes { inputPtr in
            output.withUnsafeMutableBytes { outputPtr in
                var src = vImage_Buffer(
                    data: inputPtr.baseAddress,
                    height: 1,
                    width: UInt(totalCount),
                    rowBytes: MemoryLayout<Float32>.size * totalCount
                )
                var dst = vImage_Buffer(
                    data: outputPtr.baseAddress,
                    height: 1,
                    width: UInt(totalCount),
                    rowBytes: MemoryLayout<UInt16>.size * totalCount
                )
                
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, .zero)
            }
        }
        
        return output
    }
    
    init(fp16Array: [UInt16]) {
        self.init(repeating: 0.0, count: fp16Array.count)
        var fp16ArrayCopy = fp16Array
        let totalCount = fp16Array.count
        fp16ArrayCopy.withUnsafeMutableBytes { inputPtr in
            self.withUnsafeMutableBytes { outputPtr in
                var src = vImage_Buffer(
                    data: inputPtr.baseAddress,
                    height: 1,
                    width: UInt(totalCount),
                    rowBytes: MemoryLayout<UInt16>.size * totalCount
                )
                var dst = vImage_Buffer(
                    data: outputPtr.baseAddress,
                    height: 1,
                    width: UInt(totalCount),
                    rowBytes: MemoryLayout<Float32>.size * totalCount
                )
                
                vImageConvert_Planar16FtoPlanarF(&src, &dst, .zero)
            }
        }
    }
    
}


extension Array where Element == UInt16 {
    
    mutating func toFP32() -> [Float32] {
        var output = [Float32](repeating: 0, count: self.count)
        let totalCount = count
        self.withUnsafeMutableBytes { inputPtr in
            output.withUnsafeMutableBytes { outputPtr in
                var src = vImage_Buffer(
                    data: inputPtr.baseAddress,
                    height: 1,
                    width: UInt(totalCount),
                    rowBytes: MemoryLayout<UInt16>.size * totalCount
                )
                var dst = vImage_Buffer(
                    data: outputPtr.baseAddress,
                    height: 1,
                    width: UInt(totalCount),
                    rowBytes: MemoryLayout<Float32>.size * totalCount
                )
                
                vImageConvert_Planar16FtoPlanarF(&src, &dst, .zero)
            }
        }
        
        return output
    }
    
    init(fp32Array: [Float32]) {
        self.init(repeating: 0, count: fp32Array.count)
        var fp32ArrayCopy = fp32Array
        let totalCount = fp32Array.count
        fp32ArrayCopy.withUnsafeMutableBytes { inputPtr in
            self.withUnsafeMutableBytes { outputPtr in
                var src = vImage_Buffer(
                    data: inputPtr.baseAddress,
                    height: 1,
                    width: UInt(totalCount),
                    rowBytes: MemoryLayout<Float32>.size * totalCount
                )
                var dst = vImage_Buffer(
                    data: outputPtr.baseAddress,
                    height: 1,
                    width: UInt(totalCount),
                    rowBytes: MemoryLayout<UInt16>.size * totalCount
                )
                
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, .zero)
            }
        }
    }
    
}
