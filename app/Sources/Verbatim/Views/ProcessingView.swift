import SwiftUI
import VerbatimCore

struct ProcessingView: View {
    let onOpen: (String) -> Void
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Group {
            if model.activeJobs.isEmpty && model.recentlyFinishedJobs.isEmpty {
                EmptyState(title: "Nothing is processing", systemImage: "waveform",
                           message: "Start a transcription from the Library.")
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if !model.activeJobs.isEmpty {
                            Text("In progress").font(.headline)
                            ForEach(model.activeJobs) { job in
                                JobCard(job: job, onOpen: onOpen)
                            }
                        }
                        if !model.recentlyFinishedJobs.isEmpty {
                            Text("Recently finished").font(.headline).padding(.top, 8)
                            ForEach(model.recentlyFinishedJobs) { job in
                                JobCard(job: job, onOpen: onOpen, compact: true)
                            }
                        }
                    }
                    .padding(20)
                    .frame(maxWidth: 820, alignment: .leading)
                }
            }
        }
        .navigationTitle("Processing")
    }
}

struct JobCard: View {
    let job: Job
    let onOpen: (String) -> Void
    var compact = false
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(job.projectTitle ?? "Untitled").font(.body.weight(.semibold))
                    Text(summary).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                JobElapsed(job: job)
                actions
            }
            if !compact {
                JobStagesView(job: job)
            } else if let error = job.error?.message, job.status != .completed {
                Text(error).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.secondary.opacity(0.15)))
    }

    private var summary: String {
        let mode = model.modes.first { $0.id == job.mode }?.label ?? job.mode
        switch job.status {
        case .queued: return "\(mode) · waiting for the current job to finish"
        case .running: return "\(mode) · attempt \(job.attempt)"
        case .completed: return "\(mode) · completed"
        case .failed: return "\(mode) · failed"
        case .cancelled: return "\(mode) · cancelled"
        case .interrupted: return "\(mode) · interrupted"
        }
    }

    @ViewBuilder
    private var actions: some View {
        HStack(spacing: 6) {
            if job.status.isActive {
                Button("Cancel") { Task { await model.cancel(job) } }
                    .disabled(job.cancelRequested)
            }
            if job.status.isResumable {
                Button("Resume") { Task { await model.resume(job) } }
            }
            Button("Open") { onOpen(job.projectId) }
        }
        .controlSize(.small)
    }
}

/// Elapsed wall-clock time of a running job, ticking once per second.
struct JobElapsed: View {
    let job: Job

    var body: some View {
        if job.status == .running, let started = ISO8601.parse(job.startedAt) {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                Text(Timecode.clock(context.date.timeIntervalSince(started)))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        } else if let elapsed = job.elapsedSeconds, job.status != .queued {
            Text(Timecode.clock(elapsed)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
    }
}

struct JobStagesView: View {
    let job: Job

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(job.stages) { stage in
                StageRow(stage: stage, jobRunning: job.status == .running)
            }
            if let error = job.error, job.status != .completed {
                Label(error.message ?? "Unknown error", systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(job.status == .failed ? .red : .orange)
                    .textSelection(.enabled)
            }
        }
    }
}

struct StageRow: View {
    let stage: JobStage
    let jobRunning: Bool

    var body: some View {
        HStack(spacing: 8) {
            icon.frame(width: 16)
            Text(stage.label).font(.callout).frame(width: 140, alignment: .leading)
            if stage.status == .running {
                // Determinate only when the engine measured real progress.
                if let fraction = stage.fraction {
                    ProgressView(value: fraction).frame(maxWidth: 260)
                    Text(fraction.formatted(.percent.precision(.fractionLength(0))))
                        .font(.caption.monospacedDigit())
                        .frame(width: 38, alignment: .trailing)
                } else {
                    ProgressView().controlSize(.small)
                }
                if let eta = stage.etaSeconds {
                    Text("about \(Timecode.duration(eta)) left").font(.caption).foregroundStyle(.secondary)
                }
            } else if let detail = stage.detail {
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private var icon: some View {
        switch stage.status {
        case .pending: Image(systemName: "circle").foregroundStyle(.tertiary)
        case .running: Image(systemName: "circle.dotted").foregroundStyle(Color.accentColor)
        case .completed: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed: Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
        case .cancelled: Image(systemName: "stop.circle").foregroundStyle(.orange)
        case .skipped: Image(systemName: "minus.circle").foregroundStyle(.secondary)
        }
    }
}
