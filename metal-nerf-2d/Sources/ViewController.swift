//
//  ViewController.swift
//  metal-nerf-2d
//
//  Created by Vyacheslav Gilevich on 14.08.2022.
//

import UIKit

class ViewController: UIViewController {
    
    private lazy var image = UIImage(named: "512px")
    private lazy var nerf = NERF2DMPSNN(batchSize: self.image?.cgImage?.width ?? 512, networkWidth: 512, networkDepth: 3)
    
    private lazy var sourceImageView = UIImageView(image: self.image)
    private lazy var targetImageView = UIImageView()

    override func viewDidLoad() {
        super.viewDidLoad()
        
        sourceImageView.contentMode = .scaleAspectFit
        view.addSubview(sourceImageView)
        
        sourceImageView.translatesAutoresizingMaskIntoConstraints = false
        
        sourceImageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor).isActive = true
        sourceImageView.leftAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leftAnchor).isActive = true
        sourceImageView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor).isActive = true
        
        targetImageView.contentMode = .scaleAspectFit
        view.addSubview(targetImageView)
        
        targetImageView.translatesAutoresizingMaskIntoConstraints = false
        
        targetImageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor).isActive = true
        targetImageView.leftAnchor.constraint(equalTo: sourceImageView.rightAnchor).isActive = true
        targetImageView.rightAnchor.constraint(equalTo: view.safeAreaLayoutGuide.rightAnchor).isActive = true
        targetImageView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor).isActive = true
        targetImageView.widthAnchor.constraint(equalTo: sourceImageView.widthAnchor).isActive = true
        
    }
    
    override func viewDidAppear(_ animated: Bool) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let image = self?.image else { return }
            self?.nerf?.train(on: image, epochs: 8192) { image in
                DispatchQueue.main.async { [weak self] in
                    self?.targetImageView.image = image
                }
            }
        }
    }


}

