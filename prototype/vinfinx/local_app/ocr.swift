import Foundation
import Vision
import AppKit
import PDFKit

guard CommandLine.arguments.count == 2 else { exit(2) }
let url = URL(fileURLWithPath: CommandLine.arguments[1])
var results: [[String: Any]] = []
func recognize(_ image: CGImage, _ page: Int) throws {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = true
    try VNImageRequestHandler(cgImage: image).perform([request])
    for item in request.results ?? [] {
        if let text = item.topCandidates(1).first {
            results.append(["text": text.string, "confidence": text.confidence, "page": page,
                            "box": [item.boundingBox.origin.x, item.boundingBox.origin.y, item.boundingBox.width, item.boundingBox.height]])
        }
    }
}
do {
    if url.pathExtension.lowercased() == "pdf" {
        guard let pdf = PDFDocument(url: url), pdf.pageCount <= 5 else { exit(3) }
        for i in 0..<pdf.pageCount {
            guard let page = pdf.page(at:i) else { continue }
            let img=page.thumbnail(of:NSSize(width:1800,height:2400),for:.mediaBox)
            var rect=CGRect(origin:.zero,size:img.size)
            if let cg=img.cgImage(forProposedRect:&rect,context:nil,hints:nil) { try recognize(cg,i+1) }
        }
    } else if let image=NSImage(contentsOf:url) {
        var rect=CGRect(origin:.zero,size:image.size)
        if let cg=image.cgImage(forProposedRect:&rect,context:nil,hints:nil) { try recognize(cg,1) }
    }
    let data=try JSONSerialization.data(withJSONObject:results)
    print(String(data:data,encoding:.utf8)!)
} catch { fputs("Local OCR failed\n",stderr); exit(4) }
