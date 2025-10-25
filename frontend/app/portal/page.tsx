"use client"

import { useState } from "react"
import Link from "next/link"

// Mock data for company's ad history
const mockAdHistory = [
  {
    id: 1,
    name: "Summer Collection Launch",
    date: "2025-10-15",
    biasScore: 92,
    status: "approved",
    category: "Fashion",
    issues: ["Minor gendered language in headline"],
  },
  {
    id: 2,
    name: "Tech Product Demo",
    date: "2025-10-10",
    biasScore: 78,
    status: "revised",
    category: "Technology",
    issues: ["Lack of diversity in visuals", "Age stereotyping detected"],
  },
  {
    id: 3,
    name: "Holiday Special Offer",
    date: "2025-10-05",
    biasScore: 95,
    status: "approved",
    category: "Retail",
    issues: [],
  },
  {
    id: 4,
    name: "Fitness App Campaign",
    date: "2025-09-28",
    biasScore: 65,
    status: "rejected",
    category: "Health & Wellness",
    issues: ["Body shaming implications", "Gender stereotyping", "Exclusionary language"],
  },
  {
    id: 5,
    name: "Back to School",
    date: "2025-09-20",
    biasScore: 88,
    status: "approved",
    category: "Education",
    issues: ["Minor accessibility concerns"],
  },
]

const companyStats = {
  totalAdsScanned: 47,
  averageBiasScore: 84,
  adsApproved: 35,
  adsRevised: 8,
  adsRejected: 4,
  improvementRate: "+12%",
}

// Mock data for bias score trend over time
const biasScoreTrend = [
  { date: "Sep 1", score: 76 },
  { date: "Sep 8", score: 78 },
  { date: "Sep 15", score: 74 },
  { date: "Sep 22", score: 80 },
  { date: "Sep 29", score: 82 },
  { date: "Oct 6", score: 85 },
  { date: "Oct 13", score: 83 },
  { date: "Oct 20", score: 88 },
  { date: "Oct 27", score: 91 },
]

function StatCard({ title, value, subtitle }: { title: string; value: string | number; subtitle?: string }) {
  return (
    <div className="bg-white border border-border rounded-lg p-6 flex flex-col gap-2 shadow-sm">
      <div className="text-muted-foreground text-sm font-medium">{title}</div>
      <div className="text-foreground text-3xl font-bold">{value}</div>
      {subtitle && <div className="text-xs text-green-600 font-medium">{subtitle}</div>}
    </div>
  )
}

function BiasScoreChart({ data }: { data: typeof biasScoreTrend }) {
  const width = 800
  const height = 250
  const padding = { top: 20, right: 20, bottom: 40, left: 50 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const minScore = 60
  const maxScore = 100

  // Calculate points for the line
  const points = data.map((item, index) => {
    const x = padding.left + (index / (data.length - 1)) * chartWidth
    const y = padding.top + chartHeight - ((item.score - minScore) / (maxScore - minScore)) * chartHeight
    return { x, y, ...item }
  })

  // Create path for the line
  const linePath = points.map((point, index) => {
    if (index === 0) return `M ${point.x} ${point.y}`
    return `L ${point.x} ${point.y}`
  }).join(' ')

  // Create area path
  const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`

  // Y-axis labels
  const yLabels = [60, 70, 80, 90, 100]

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        {/* Grid lines */}
        {yLabels.map((label) => {
          const y = padding.top + chartHeight - ((label - minScore) / (maxScore - minScore)) * chartHeight
          return (
            <g key={label}>
              <line
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke="#e5e7eb"
                strokeWidth="1"
              />
              <text x={padding.left - 10} y={y + 4} textAnchor="end" className="text-xs fill-gray-500">
                {label}
              </text>
            </g>
          )
        })}

        {/* Area gradient */}
        <defs>
          <linearGradient id="areaGradient" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.05" />
          </linearGradient>
        </defs>

        {/* Area under the line */}
        <path d={areaPath} fill="url(#areaGradient)" />

        {/* Line */}
        <path d={linePath} fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />

        {/* Data points */}
        {points.map((point, index) => (
          <g key={index}>
            <circle cx={point.x} cy={point.y} r="5" fill="white" stroke="#10b981" strokeWidth="3" />
            <circle cx={point.x} cy={point.y} r="2" fill="#10b981" />
          </g>
        ))}

        {/* X-axis labels */}
        {points.map((point, index) => (
          <text
            key={index}
            x={point.x}
            y={height - padding.bottom + 20}
            textAnchor="middle"
            className="text-xs fill-gray-500"
          >
            {point.date}
          </text>
        ))}

        {/* Score labels on hover */}
        {points.map((point, index) => (
          <text
            key={`score-${index}`}
            x={point.x}
            y={point.y - 15}
            textAnchor="middle"
            className="text-xs font-semibold fill-gray-700"
          >
            {point.score}
          </text>
        ))}
      </svg>
    </div>
  )
}

function AdHistoryRow({ ad }: { ad: typeof mockAdHistory[0] }) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "approved":
        return "bg-green-100 text-green-800"
      case "revised":
        return "bg-yellow-100 text-yellow-800"
      case "rejected":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 90) return "text-green-600"
    if (score >= 75) return "text-yellow-600"
    return "text-red-600"
  }

  return (
    <div className="bg-white border border-border rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="font-semibold text-foreground">{ad.name}</h3>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(ad.status)}`}>
              {ad.status.charAt(0).toUpperCase() + ad.status.slice(1)}
            </span>
          </div>
          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
            <span>📅 {new Date(ad.date).toLocaleDateString()}</span>
            <span>📂 {ad.category}</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-center">
            <div className="text-xs text-muted-foreground mb-1">Bias Score</div>
            <div className={`text-2xl font-bold ${getScoreColor(ad.biasScore)}`}>{ad.biasScore}</div>
          </div>
        </div>
      </div>
      {ad.issues.length > 0 && (
        <div className="mt-3 pt-3 border-t border-border">
          <div className="text-xs font-medium text-muted-foreground mb-2">Issues Detected:</div>
          <div className="flex flex-wrap gap-2">
            {ad.issues.map((issue, idx) => (
              <span key={idx} className="px-2 py-1 bg-red-50 text-red-700 rounded text-xs">
                {issue}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function PortalPage() {
  const [timeRange, setTimeRange] = useState("30d")

  return (
    <div className="min-h-screen bg-background pt-12 sm:pt-14 md:pt-16 lg:pt-[84px]">
      {/* Header */}
      <div className="bg-white border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground">Company Portal</h1>
              <p className="text-muted-foreground mt-1">Acme Corporation</p>
            </div>
            <Link
              href="/upload"
              className="px-6 py-2 bg-primary text-primary-foreground rounded-full font-medium hover:opacity-90 transition-opacity"
            >
              New Audit
            </Link>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Grid */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-foreground">Overview</h2>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-1.5 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
              <option value="1y">Last year</option>
            </select>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <StatCard title="Total Ads Scanned" value={companyStats.totalAdsScanned} />
            <StatCard
              title="Average Bias Score"
              value={companyStats.averageBiasScore}
              subtitle={companyStats.improvementRate + " from last month"}
            />
            <StatCard title="Ads Approved" value={companyStats.adsApproved} />
            <StatCard title="Ads Revised" value={companyStats.adsRevised} />
            <StatCard title="Ads Rejected" value={companyStats.adsRejected} />
            <StatCard
              title="Success Rate"
              value={`${Math.round((companyStats.adsApproved / companyStats.totalAdsScanned) * 100)}%`}
            />
          </div>
        </div>

        {/* Bias Score Trend Chart */}
        <div className="mb-8 bg-white border border-border rounded-lg p-6">
          <h2 className="text-xl font-semibold text-foreground mb-4">Bias Score Trend</h2>
          <BiasScoreChart data={biasScoreTrend} />
        </div>

        {/* Ad History */}
        <div>
          <h2 className="text-xl font-semibold text-foreground mb-4">Recent Ad Audits</h2>
          <div className="space-y-4">
            {mockAdHistory.map((ad) => (
              <AdHistoryRow key={ad.id} ad={ad} />
            ))}
          </div>
        </div>

        {/* Empty State for no data */}
        {mockAdHistory.length === 0 && (
          <div className="bg-white border border-border rounded-lg p-12 text-center">
            <div className="text-muted-foreground">
              <svg
                className="mx-auto h-12 w-12 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <h3 className="text-lg font-medium text-foreground mb-2">No ads audited yet</h3>
              <p className="mb-4">Start by uploading your first ad for bias detection</p>
              <Link
                href="/upload"
                className="inline-block px-6 py-2 bg-primary text-primary-foreground rounded-full font-medium hover:opacity-90 transition-opacity"
              >
                Upload First Ad
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
