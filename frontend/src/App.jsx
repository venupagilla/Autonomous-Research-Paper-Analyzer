import { useEffect, useMemo, useRef, useState } from 'react'
import { jsPDF } from 'jspdf'
import ReactMarkdown from 'react-markdown'
import './App.css'

const samplePapers = [
  {
    id: 1,
    title: 'Efficient Distillation Strategies for Domain Adaptation in LLMs',
    venue: 'ACL 2025',
    score: 91,
    summary:
      'Shows a two-stage distillation pipeline that preserves reasoning quality while reducing inference cost by 38%.',
    tags: ['LLM', 'Distillation', 'Benchmarking'],
  },
  {
    id: 2,
    title: 'Autonomous Research Agents with Verifiable Tool Traces',
    venue: 'NeurIPS 2025',
    score: 88,
    summary:
      'Introduces trace-grounded reporting for autonomous agents, improving reproducibility and reviewer confidence.',
    tags: ['Agents', 'Reproducibility', 'Evaluation'],
  },
  {
    id: 3,
    title: 'Retrieval-Augmented Survey Generation at Scale',
    venue: 'ICLR 2026',
    score: 85,
    summary:
      'Presents a retrieval strategy that clusters citations and generates concise comparative survey sections.',
    tags: ['RAG', 'NLP', 'Survey'],
  },
]

function App() {
  const POLL_INTERVAL_QUEUED_MS = 3000
  const POLL_INTERVAL_RUNNING_MS = 8000
  const POLL_INTERVAL_FINALIZING_MS = 5000

  const [query, setQuery] = useState('autonomous research agents')
  const [maxPapers, setMaxPapers] = useState(5)
  const [analysisMode, setAnalysisMode] = useState('arxiv-title')
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [activeFilter, setActiveFilter] = useState('all')
  const [papers, setPapers] = useState(samplePapers)
  const [reportMarkdown, setReportMarkdown] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [jobStatus, setJobStatus] = useState('')
  const [jobProgress, setJobProgress] = useState(0)
  const [jobStage, setJobStage] = useState('queued')
  const [jobHistory, setJobHistory] = useState([])
  const [isExportingPdf, setIsExportingPdf] = useState(false)
  const [validationWarnings, setValidationWarnings] = useState([])
  const [reportIsGrounded, setReportIsGrounded] = useState(true)

  const abortControllerRef = useRef(null)
  const activeJobIdRef = useRef('')
  const activeHistoryIdRef = useRef('')
  const isCanceledRef = useRef(false)

  const apiBaseUrl =
    import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

  const stageOrder = ['queued', 'running', 'finalizing']

  const getStageFromStatus = (status) => {
    if (status === 'running') {
      return 'running'
    }

    if (status === 'finalizing' || status === 'completed') {
      return 'finalizing'
    }

    return 'queued'
  }

  const upsertJobHistory = (jobId, changes) => {
    setJobHistory((prev) => {
      const existingIndex = prev.findIndex((item) => item.jobId === jobId)
      const entry = {
        jobId,
        topic: query,
        status: 'queued',
        progress: 0,
        updatedAt: new Date().toISOString(),
        ...changes,
      }

      if (existingIndex === -1) {
        return [entry, ...prev].slice(0, 6)
      }

      const next = [...prev]
      next[existingIndex] = { ...next[existingIndex], ...entry }
      return next
    })
  }

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  const visiblePapers = useMemo(() => {
    const source = papers

    if (activeFilter === 'all') {
      return source
    }

    return source.filter((paper) =>
      paper.tags.some((tag) => tag.toLowerCase() === activeFilter),
    )
  }, [activeFilter, papers])

  const shouldUseAsyncMode = () => {
    if (analysisMode === 'uploaded-pdfs') {
      return uploadedFiles.length >= 3 || maxPapers >= 8
    }

    const wordCount = query.trim().split(/\s+/).filter(Boolean).length
    return maxPapers >= 8 || wordCount >= 6
  }

  const buildJsonPayload = () => ({
    topic: query,
    max_papers: maxPapers,
  })

  const buildUploadFormPayload = () => {
    const formData = new FormData()
    uploadedFiles.forEach((file) => {
      formData.append('files', file)
    })

    if (query.trim()) {
      formData.append('topic', query.trim())
    }

    formData.append('max_papers', String(maxPapers))
    return formData
  }

  const clearNetworkControls = () => {
    abortControllerRef.current = null
  }

  const normalizePapers = (backendPapers) => {
    if (!Array.isArray(backendPapers)) {
      return []
    }

    return backendPapers.map((paper, index) => {
      const categories = Array.isArray(paper.categories) ? paper.categories : []
      const tags = categories.slice(0, 3).map((value) => String(value))

      return {
        id: paper.arxiv_id || index + 1,
        title: paper.title || 'Untitled paper',
        venue: paper.published || 'Unknown date',
        score: Number(paper.relevance_score) || 0,
        summary: paper.abstract_snippet || 'No summary available.',
        tags: tags.length > 0 ? tags : ['general'],
      }
    })
  }

  const applyResultToUi = (data) => {
    const normalizedPapers = normalizePapers(data.papers)
    setPapers(normalizedPapers)
    setReportMarkdown(data.report_markdown || '')
    setValidationWarnings(
      Array.isArray(data.validation_warnings) ? data.validation_warnings : [],
    )
    setReportIsGrounded(data.report_is_grounded !== false)
    setActiveFilter('all')
  }

  const buildExportFileName = (extension) => {
    const slug = query
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .trim()
      .replace(/\s+/g, '-')
      .slice(0, 60)

    const safeSlug = slug || 'research-report'
    return `${safeSlug}.${extension}`
  }

  const exportMarkdown = () => {
    if (!reportMarkdown.trim()) {
      setError('No report available to export yet.')
      return
    }

    const blob = new Blob([reportMarkdown], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = buildExportFileName('md')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const exportPdf = async () => {
    if (!reportMarkdown.trim()) {
      setError('No report available to export yet.')
      return
    }

    setIsExportingPdf(true)
    setError('')

    try {
      const doc = new jsPDF({ unit: 'pt', format: 'a4' })
      const pageWidth = doc.internal.pageSize.getWidth()
      const pageHeight = doc.internal.pageSize.getHeight()
      const margin = 48
      const contentWidth = pageWidth - margin * 2
      const maxY = pageHeight - margin

      doc.setFont('helvetica', 'bold')
      doc.setFontSize(16)
      doc.text('Autonomous Research Report', margin, margin)

      doc.setFont('helvetica', 'normal')
      doc.setFontSize(10)
      doc.text(`Topic: ${query}`, margin, margin + 20)
      doc.text(`Generated: ${new Date().toLocaleString()}`, margin, margin + 34)

      doc.setFontSize(11)
      let y = margin + 58
      const lines = doc.splitTextToSize(reportMarkdown, contentWidth)

      for (const line of lines) {
        if (y > maxY) {
          doc.addPage()
          y = margin
        }

        doc.text(line, margin, y)
        y += 15
      }

      doc.save(buildExportFileName('pdf'))
    } catch (pdfError) {
      setError(pdfError.message || 'Failed to export PDF.')
    } finally {
      setIsExportingPdf(false)
    }
  }

  const viewJobDetails = async (job) => {
    setError('')

    if (job.result) {
      applyResultToUi(job.result)
      setQuery(job.topic || query)
      setJobStatus(`Loaded results from job ${job.jobId}`)
      setJobProgress(Number(job.progress) || 100)
      setJobStage('finalizing')
      return
    }

    if (String(job.jobId).startsWith('sync-')) {
      setError('This sync job has no cached result snapshot.')
      return
    }

    try {
      const response = await fetch(`${apiBaseUrl}/api/jobs/${job.jobId}`)
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || 'Failed to load job details')
      }

      const data = await response.json()
      if (data.status !== 'completed' || !data.result) {
        throw new Error('Job result is not available yet')
      }

      applyResultToUi(data.result)
      setQuery(job.topic || query)
      setJobStatus(`Loaded results from job ${job.jobId}`)
      setJobProgress(100)
      setJobStage('finalizing')

      upsertJobHistory(job.jobId, {
        status: data.status,
        progress: Number(data.progress) || 100,
        updatedAt: data.updated_at || new Date().toISOString(),
        result: data.result,
      })
    } catch (detailsError) {
      setError(detailsError.message || 'Unable to load job details')
    }
  }

  const pollAsyncJob = async (jobId, signal) => {
    while (true) {
      if (isCanceledRef.current) {
        throw new Error('Request canceled by user')
      }

      const response = await fetch(`${apiBaseUrl}/api/jobs/${jobId}`, { signal })
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || 'Failed to fetch async job status')
      }

      const data = await response.json()
      const progress = Number(data.progress) || 0
      setJobStatus(`Async mode: ${data.status} (${progress}%)`)
      setJobProgress(progress)
      setJobStage(getStageFromStatus(data.status))
      upsertJobHistory(jobId, {
        status: data.status,
        progress,
        updatedAt: data.updated_at || new Date().toISOString(),
      })

      if (data.status === 'completed') {
        upsertJobHistory(jobId, {
          status: 'completed',
          progress: Number(data.progress) || 100,
          updatedAt: data.updated_at || new Date().toISOString(),
          result: data.result || {},
        })
        return data.result || {}
      }

      if (data.status === 'failed') {
        throw new Error(data.message || 'Async analysis failed')
      }

      if (data.status === 'canceled') {
        throw new Error('Request canceled by user')
      }

      let waitMs = POLL_INTERVAL_RUNNING_MS
      if (data.status === 'queued') {
        waitMs = POLL_INTERVAL_QUEUED_MS
      } else if (data.status === 'finalizing') {
        waitMs = POLL_INTERVAL_FINALIZING_MS
      }

      await new Promise((resolve) => {
        setTimeout(resolve, waitMs)
      })
    }
  }

  const startSyncAnalysis = async (signal) => {
    setJobStatus('Sync mode: running analysis...')
    setJobProgress(35)
    setJobStage('running')

    const endpoint =
      analysisMode === 'uploaded-pdfs'
        ? `${apiBaseUrl}/api/analyze/upload`
        : `${apiBaseUrl}/api/analyze`

    const requestInit =
      analysisMode === 'uploaded-pdfs'
        ? {
            method: 'POST',
            body: buildUploadFormPayload(),
            signal,
          }
        : {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(buildJsonPayload()),
            signal,
          }

    const response = await fetch(endpoint, requestInit)

    if (!response.ok) {
      const data = await response.json().catch(() => ({}))
      throw new Error(data.detail || 'Failed to run analysis')
    }

    return response.json()
  }

  const startAsyncAnalysis = async (signal) => {
    setJobStatus('Async mode: queueing job...')

    const endpoint =
      analysisMode === 'uploaded-pdfs'
        ? `${apiBaseUrl}/api/analyze/upload/async`
        : `${apiBaseUrl}/api/analyze/async`

    const requestInit =
      analysisMode === 'uploaded-pdfs'
        ? {
            method: 'POST',
            body: buildUploadFormPayload(),
            signal,
          }
        : {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(buildJsonPayload()),
            signal,
          }

    const kickoffResponse = await fetch(endpoint, requestInit)

    if (!kickoffResponse.ok) {
      const data = await kickoffResponse.json().catch(() => ({}))
      throw new Error(data.detail || 'Failed to start async analysis')
    }

    const kickoffPayload = await kickoffResponse.json()
    const jobId = String(kickoffPayload.job_id || '')
    if (!jobId) {
      throw new Error('Async analysis did not return a job id')
    }

    activeJobIdRef.current = jobId
    activeHistoryIdRef.current = jobId
    setJobProgress(5)
    setJobStage('queued')
    upsertJobHistory(jobId, {
      status: 'queued',
      progress: 0,
      updatedAt: new Date().toISOString(),
    })
    return pollAsyncJob(jobId, signal)
  }

  const cancelAnalysis = async () => {
    isCanceledRef.current = true

    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    clearNetworkControls()

    if (activeJobIdRef.current) {
      try {
        await fetch(`${apiBaseUrl}/api/jobs/${activeJobIdRef.current}/cancel`, {
          method: 'POST',
        })
      } catch {
        // Ignore cancel propagation errors.
      }
    }

    setIsLoading(false)
    setJobStatus('Canceled by user')
    setJobProgress(0)

    if (activeHistoryIdRef.current) {
      upsertJobHistory(activeHistoryIdRef.current, {
        status: 'canceled',
        progress: 0,
        updatedAt: new Date().toISOString(),
      })
    }
  }

  const runAnalysis = async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    isCanceledRef.current = false
    activeJobIdRef.current = ''
    activeHistoryIdRef.current = ''
    setIsLoading(true)
    setError('')
    setJobStatus('')
    setJobProgress(0)
    setJobStage('queued')

    const trimmedQuery = query.trim()
    if (analysisMode === 'arxiv-title' && trimmedQuery.length < 3) {
      setError('Please enter at least 3 characters for topic/title mode.')
      setIsLoading(false)
      return
    }

    if (analysisMode === 'uploaded-pdfs' && uploadedFiles.length === 0) {
      setError('Please upload at least one PDF file in uploaded mode.')
      setIsLoading(false)
      return
    }

    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      const useAsyncMode = shouldUseAsyncMode()

      if (!useAsyncMode) {
        const syncJobId = `sync-${Date.now()}`
        activeHistoryIdRef.current = syncJobId
        upsertJobHistory(syncJobId, {
          status: 'running',
          progress: 35,
          updatedAt: new Date().toISOString(),
          topic: query,
        })
      }

      const data = useAsyncMode
        ? await startAsyncAnalysis(controller.signal)
        : await startSyncAnalysis(controller.signal)

      applyResultToUi(data)
      setJobStatus('Completed')
      setJobProgress(100)
      setJobStage('finalizing')
      if (activeHistoryIdRef.current) {
        upsertJobHistory(activeHistoryIdRef.current, {
          status: 'completed',
          progress: 100,
          updatedAt: new Date().toISOString(),
          result: data,
        })
      }
    } catch (requestError) {
      if (isCanceledRef.current) {
        setError('')
      } else {
        setError(requestError.message || 'Request failed')
        if (activeHistoryIdRef.current) {
          upsertJobHistory(activeHistoryIdRef.current, {
            status: 'failed',
            progress: jobProgress,
            updatedAt: new Date().toISOString(),
          })
        }
      }
    } finally {
      clearNetworkControls()
      setIsLoading(false)
      activeJobIdRef.current = ''
      activeHistoryIdRef.current = ''
    }
  }

  return (
    <div className="page-shell">
      <header className="hero-panel">
        <p className="eyebrow">Autonomous Research Paper Analyzer</p>
        <h1>Extract insights with intelligent research analysis.</h1>
        <p className="hero-copy">
          A React frontend for running your analysis pipeline, tracking key
          findings, and preparing evidence-backed research briefs.
        </p>

        <form
          className="search-row"
          onSubmit={(event) => {
            event.preventDefault()
            runAnalysis()
          }}
        >
          <select
            value={analysisMode}
            onChange={(event) => setAnalysisMode(event.target.value)}
            aria-label="Analysis mode"
            disabled={isLoading}
          >
            <option value="arxiv-title">arXiv Title Mode</option>
            <option value="uploaded-pdfs">Uploaded PDFs Mode</option>
          </select>
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={
              analysisMode === 'uploaded-pdfs'
                ? 'Optional topic hint (e.g., kidney tumor segmentation)'
                : 'Search topic, method, or venue'
            }
            aria-label="Research query"
          />
          {analysisMode === 'uploaded-pdfs' ? (
            <input
              type="file"
              accept="application/pdf,.pdf"
              multiple
              onChange={(event) => {
                const files = Array.from(event.target.files || [])
                setUploadedFiles(files)
              }}
              disabled={isLoading}
              aria-label="Upload PDF files"
              className="upload-input"
            />
          ) : null}
          <input
            type="number"
            min="1"
            max="15"
            value={maxPapers}
            onChange={(event) => setMaxPapers(Number(event.target.value) || 5)}
            aria-label="Max papers"
            className="max-papers"
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </form>

        {analysisMode === 'uploaded-pdfs' ? (
          <p className="upload-hint">
            Upload 1 to 10 PDF files. Topic is optional in this mode.
            {uploadedFiles.length > 0 ? ` Selected: ${uploadedFiles.length}` : ''}
          </p>
        ) : null}

        {isLoading ? (
          <button type="button" className="cancel-btn" onClick={cancelAnalysis}>
            Cancel
          </button>
        ) : null}

        {error ? <p className="status-message error">{error}</p> : null}
        {!error && isLoading ? (
          <p className="status-message">
            {analysisMode === 'uploaded-pdfs'
              ? 'Parsing uploaded PDFs and generating report...'
              : 'Fetching papers and generating report...'}
          </p>
        ) : null}
        {!error && jobStatus ? <p className="status-message">{jobStatus}</p> : null}

        {(isLoading || jobProgress > 0) ? (
          <div className="progress-wrap" aria-live="polite">
            <div className="stage-labels">
              {stageOrder.map((stage) => {
                const currentIndex = stageOrder.indexOf(jobStage)
                const stageIndex = stageOrder.indexOf(stage)
                const className =
                  stageIndex <= currentIndex ? 'stage-chip is-active' : 'stage-chip'

                return (
                  <span key={stage} className={className}>
                    {stage[0].toUpperCase() + stage.slice(1)}
                  </span>
                )
              })}
            </div>
            <div className="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={jobProgress}>
              <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, jobProgress))}%` }} />
            </div>
          </div>
        ) : null}

        <div className="metric-strip">
          <article>
            <h2>{papers.length}</h2>
            <p>Papers scanned</p>
          </article>
          <article>
            <h2>{papers.filter((paper) => paper.score >= 75).length}</h2>
            <p>High-relevance matches</p>
          </article>
          <article>
            <h2>{maxPapers}</h2>
            <p>Max papers requested</p>
          </article>
        </div>
      </header>

      <main className="content-grid">
        <section className="results-panel">
          <div className="panel-head">
            <h3>Top Results</h3>
            <p>Query: {query}</p>
          </div>

          <div className="filter-row" role="tablist" aria-label="Paper filters">
            {['all', 'agents', 'rag', 'benchmarking'].map((filter) => (
              <button
                key={filter}
                type="button"
                className={filter === activeFilter ? 'chip is-active' : 'chip'}
                onClick={() => setActiveFilter(filter)}
              >
                {filter}
              </button>
            ))}
          </div>

          <ul className="paper-list">
            {visiblePapers.map((paper) => (
              <li key={paper.id} className="paper-card">
                <div>
                  <h4>{paper.title}</h4>
                  <p className="venue">{paper.venue}</p>
                  <p>{paper.summary}</p>
                  <div className="tag-row">
                    {paper.tags.map((tag) => (
                      <span key={tag}>{tag}</span>
                    ))}
                  </div>
                </div>
                <p className="score">{paper.score}</p>
              </li>
            ))}
            {visiblePapers.length === 0 ? (
              <li className="paper-card empty-state">No papers returned for this filter.</li>
            ) : null}
          </ul>
        </section>

        <aside className="side-panel">
          <article>
            <h3>Workflow</h3>
            <ol>
              <li>Ingest sources and metadata</li>
              <li>Run autonomous ranking and analysis</li>
              <li>Export summary report with citations</li>
            </ol>
          </article>

          <article>
            <h3>Report Preview</h3>
            <div className="action-list">
              {validationWarnings.length > 0 ? (
                <div className="validation-warning-box" role="alert">
                  <p className="validation-warning-title">Grounding warnings</p>
                  <ul>
                    {validationWarnings.map((warning, index) => (
                      <li key={`${warning}-${index}`}>{warning}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {!reportIsGrounded ? (
                <p className="ungrounded-note">
                  This report is not fully grounded in fetched evidence.
                </p>
              ) : null}

              <div className="export-actions">
                <button type="button" onClick={exportMarkdown} disabled={isLoading}>
                  Export .md
                </button>
                <button
                  type="button"
                  onClick={exportPdf}
                  disabled={isLoading || isExportingPdf}
                >
                  {isExportingPdf ? 'Exporting PDF...' : 'Export PDF'}
                </button>
              </div>
              <div className="report-preview markdown-report">
                {reportMarkdown ? (
                  <ReactMarkdown>{reportMarkdown}</ReactMarkdown>
                ) : (
                  <p>Run analysis to generate report markdown.</p>
                )}
              </div>
            </div>
          </article>

          <article>
            <h3>Job History</h3>
            {jobHistory.length === 0 ? (
              <p className="job-history-empty">No jobs yet.</p>
            ) : (
              <ul className="job-history-list">
                {jobHistory.map((job) => (
                  <li key={job.jobId} className="job-history-item">
                    <p className="job-topic">{job.topic}</p>
                    <p className="job-meta">Status: {job.status} | {job.progress}%</p>
                    {job.status === 'completed' ? (
                      <button
                        type="button"
                        className="job-view-btn"
                        onClick={() => viewJobDetails(job)}
                      >
                        View Details
                      </button>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
          </article>
        </aside>
      </main>

      <footer className="app-footer">
        <p>Built with React + Vite for autonomous literature analysis.</p>
      </footer>
    </div>
  )
}

export default App
