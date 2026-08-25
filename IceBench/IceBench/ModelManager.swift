import Foundation

@MainActor
final class ModelManager: NSObject, ObservableObject {
    static let presets: [(name: String, url: String)] = [
        ("Qwen3.5-0.8B Q4_K_M", "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf"),
        ("Qwen3.5-0.8B Q8_0", "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"),
    ]

    @Published var downloadedModels: [URL] = []
    @Published var downloadProgress: Double? = nil
    @Published var downloadStatus: String = ""
    @Published var customURL: String = ""

    private var session: URLSession!
    private var activeTask: URLSessionDownloadTask?
    private var activeDestination: URL?

    static var modelsDir: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    override init() {
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        refresh()
    }

    func refresh() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        var found: [URL] = []
        for dir in [Self.modelsDir, docs] {
            if let items = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) {
                found += items.filter { $0.pathExtension.lowercased() == "gguf" }
            }
        }
        downloadedModels = found.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    func download(urlString: String) {
        guard activeTask == nil else { return }
        guard let url = URL(string: urlString.trimmingCharacters(in: .whitespacesAndNewlines)),
              url.scheme?.hasPrefix("http") == true else {
            downloadStatus = "Invalid URL"
            return
        }
        let filename = url.lastPathComponent.isEmpty ? "model.gguf" : url.lastPathComponent
        activeDestination = Self.modelsDir.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: activeDestination!.path) {
            downloadStatus = "\(filename) already downloaded"
            activeDestination = nil
            refresh()
            return
        }
        downloadStatus = "Downloading \(filename)…"
        downloadProgress = 0
        let task = session.downloadTask(with: url)
        activeTask = task
        task.resume()
    }

    func cancelDownload() {
        activeTask?.cancel()
        activeTask = nil
        downloadProgress = nil
        downloadStatus = "Download cancelled"
    }

    func delete(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
        refresh()
    }
}

extension ModelManager: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        let progress = totalBytesExpectedToWrite > 0
            ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) : 0
        let mb = Double(totalBytesWritten) / 1_048_576
        let totalMB = Double(totalBytesExpectedToWrite) / 1_048_576
        Task { @MainActor in
            self.downloadProgress = progress
            self.downloadStatus = String(format: "Downloading… %.0f / %.0f MB", mb, totalMB)
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        // Move synchronously — the temp file is deleted when this returns.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".gguf")
        try? FileManager.default.moveItem(at: location, to: tmp)
        Task { @MainActor in
            guard let dest = self.activeDestination else { return }
            do {
                try FileManager.default.moveItem(at: tmp, to: dest)
                self.downloadStatus = "Downloaded \(dest.lastPathComponent)"
            } catch {
                self.downloadStatus = "Save failed: \(error.localizedDescription)"
            }
            self.downloadProgress = nil
            self.activeTask = nil
            self.activeDestination = nil
            self.refresh()
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error, (error as NSError).code != NSURLErrorCancelled else { return }
        Task { @MainActor in
            self.downloadStatus = "Download failed: \(error.localizedDescription)"
            self.downloadProgress = nil
            self.activeTask = nil
            self.activeDestination = nil
        }
    }
}
