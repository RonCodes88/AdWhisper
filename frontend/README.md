# AdWhisper Frontend

Modern Next.js frontend for the AdWhisper ad bias detection system.

## 🎯 Features

- **Upload Interface**: Drag-and-drop file uploads with text content input
- **Real-time Analysis**: Live status tracking with progress indicators
- **Results Dashboard**: Comprehensive bias reports with:
  - Overall bias score (0-10 scale)
  - Score breakdown (text, visual, intersectional)
  - Categorized bias issues (critical, high, medium, low)
  - Actionable recommendations
  - Benchmark comparison with similar ads
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Type-Safe**: Full TypeScript implementation with strict types

## 🛠️ Tech Stack

- **Framework**: Next.js 15+ (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Hooks (useState, useEffect)
- **Routing**: Next.js App Router with dynamic routes
- **API Client**: Fetch API with custom wrapper

## 📁 Project Structure

```
frontend/
├── app/
│   ├── (landing)/
│   │   └── page.tsx              # Landing page with features showcase
│   ├── upload/
│   │   └── page.tsx              # File upload and ad submission page
│   ├── results/
│   │   └── [requestId]/
│   │       └── page.tsx          # Dynamic results page with analysis report
│   ├── components/
│   │   ├── navbar.tsx            # Navigation component
│   │   ├── cta-section.tsx       # Call-to-action section
│   │   ├── LoadingSpinner.tsx    # Reusable loading spinner
│   │   ├── StatusTracker.tsx     # Analysis progress tracker
│   │   ├── ScoreCard.tsx         # Animated score display
│   │   ├── BiasIssueCard.tsx     # Individual bias issue card
│   │   ├── RecommendationCard.tsx # Recommendation display card
│   │   └── ErrorBoundary.tsx     # Error boundary component
│   ├── layout.tsx                # Root layout
│   └── globals.css               # Global styles
├── lib/
│   └── api.ts                    # API client with all backend calls
├── types/
│   └── api.ts                    # TypeScript type definitions
├── public/                       # Static assets
├── env.example                   # Environment variables template
└── package.json                  # Dependencies and scripts
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Backend API running (see `/backend/README.md`)

### Installation

1. **Install dependencies:**

```bash
cd frontend
npm install
```

2. **Set up environment variables:**

Copy `env.example` to `.env.local`:

```bash
cp env.example .env.local
```

Update `.env.local` with your backend API URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. **Start the development server:**

```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

### Production Build

```bash
npm run build
npm start
```

## 📡 API Integration

The frontend communicates with the backend via REST API:

### API Client (`lib/api.ts`)

```typescript
// Submit ad for analysis
await submitAd({
  textContent: "Your ad copy...",
  imageFile: imageFile,
  videoFile: videoFile,
});

// Check analysis status
await checkStatus(requestId);

// Get complete results
await getResults(requestId);

// Check backend health
await getHealth();

// Poll until complete
await pollUntilComplete(requestId, onUpdate);
```

### Type Definitions (`types/api.ts`)

All API responses are fully typed:

- `AdSubmissionResponse`
- `AnalysisStatus`
- `BiasReport`
- `BiasIssue`
- `Recommendation`
- `ScoreBreakdown`
- `HealthResponse`

## 🎨 Components

### LoadingSpinner

```tsx
<LoadingSpinner size="md" text="Processing..." />
```

### StatusTracker

```tsx
<StatusTracker currentStage="analyzing_text" />
```

Stages: `submitted` → `ingestion` → `analyzing_text/analyzing_visual` → `scoring` → `complete`

### ScoreCard

```tsx
<ScoreCard score={7.5} showAnimation={true} />
```

Visual representation of bias score with color-coding:

- 0-3: Red (Critical)
- 4-6: Orange (Warning)
- 7-8: Yellow (Caution)
- 9-9.5: Green (Good)
- 9.5-10: Blue (Excellent)

### BiasIssueCard

```tsx
<BiasIssueCard issue={biasIssue} />
```

Displays individual bias issues with severity, confidence, and examples.

### RecommendationCard

```tsx
<RecommendationCard recommendation={recommendation} index={0} />
```

Shows actionable recommendations with priority and impact.

## 🔄 User Flow

1. **Landing Page** (`/`)

   - User learns about features
   - Clicks "Try for free" → redirects to `/upload`

2. **Upload Page** (`/upload`)

   - User enters text content and/or uploads image/video
   - Submits for analysis
   - Receives `request_id`
   - Redirects to `/results/{request_id}`

3. **Results Page** (`/results/[requestId]`)
   - Polls backend every 5 seconds for status
   - Shows progress tracker while processing
   - Displays complete report when ready:
     - Overall score
     - Score breakdown
     - Bias issues (grouped by severity)
     - Recommendations
     - Benchmark comparison
   - Option to analyze another ad or print report

## 🎯 Features by Page

### Upload Page

- Text content textarea
- Drag-and-drop file upload
- File validation (type, size)
- File preview with remove option
- Loading states during submission
- Error/success notifications
- Client-side validation

### Results Page

- Real-time status polling
- Animated progress tracker
- Visual score card with animation
- Score breakdown (text, visual, intersectional)
- Statistics overview
- Top concerns highlight
- Bias issues categorized by severity
- Prioritized recommendations
- Benchmark comparison
- Analysis metadata
- Print report functionality
- "Analyze another ad" button

## 🧪 Testing

### Manual Testing

1. **Start both backend and frontend:**

```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
python main.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

2. **Test upload flow:**

   - Navigate to `http://localhost:3000`
   - Click "Try for free"
   - Enter ad text or upload an image/video
   - Submit and verify redirect to results

3. **Test results page:**
   - Verify status tracker shows progress
   - Wait for analysis completion
   - Check all sections render correctly
   - Test "Analyze another ad" button

### File Upload Testing

Test with different file types:

- ✅ Images: JPG, PNG, GIF, WEBP
- ✅ Videos: MP4, MOV, AVI, WEBM
- ❌ Invalid types should show error
- ❌ Files > 50MB should show error

## 🔧 Configuration

### Environment Variables

| Variable              | Description          | Default                 |
| --------------------- | -------------------- | ----------------------- |
| `NEXT_PUBLIC_API_URL` | Backend API endpoint | `http://localhost:8000` |

### Polling Configuration

In `/app/results/[requestId]/page.tsx`:

```typescript
// Poll every 5 seconds
pollInterval = setInterval(poll, 5000);
```

Adjust interval as needed for your use case.

## 🎨 Styling

The app uses Tailwind CSS with custom theme configuration:

- **Colors**: Primary, secondary, muted, accent
- **Typography**: Sans-serif (Inter), Serif (Playfair Display)
- **Responsive**: Mobile-first design with breakpoints
- **Dark Mode**: Not yet implemented (future enhancement)

## 📦 Dependencies

Key dependencies:

```json
{
  "next": "^15.x",
  "react": "^19.x",
  "react-dom": "^19.x",
  "typescript": "^5.x",
  "tailwindcss": "^3.x"
}
```

No external UI libraries required - all components are custom-built.

## 🐛 Troubleshooting

### "Failed to fetch" error

**Problem**: Cannot connect to backend API

**Solution**:

1. Verify backend is running: `curl http://localhost:8000/health`
2. Check `.env.local` has correct `NEXT_PUBLIC_API_URL`
3. Check CORS is enabled in backend
4. Verify no firewall blocking localhost connections

### "Invalid request ID" error

**Problem**: Request ID not found in backend

**Solution**:

1. Check backend logs for request processing
2. Verify ChromaDB is initialized
3. Check backend agents are running

### File upload fails

**Problem**: File validation or upload errors

**Solution**:

1. Check file type is supported
2. Verify file size < 50MB
3. Check backend file size limits
4. Verify backend has write permissions

### Results page stuck on "Processing"

**Problem**: Polling doesn't complete

**Solution**:

1. Check backend logs for agent errors
2. Verify all backend agents are running
3. Check ChromaDB is accessible
4. Check backend for processing errors

## 🚀 Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

Set environment variables in Vercel dashboard:

- `NEXT_PUBLIC_API_URL`: Your production backend URL

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

```bash
docker build -t adwhisper-frontend .
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://api:8000 adwhisper-frontend
```

## 📝 Future Enhancements

- [ ] User authentication
- [ ] Analysis history page
- [ ] PDF report export
- [ ] Dark mode support
- [ ] Real-time WebSocket updates (replace polling)
- [ ] Chart visualizations for scores
- [ ] Comparison view for multiple ads
- [ ] Batch upload support
- [ ] Saved analyses
- [ ] Team collaboration features
- [ ] Advanced filtering and search

## 🤝 Contributing

1. Follow TypeScript strict mode
2. Use functional components with hooks
3. Maintain type safety (no `any` types)
4. Follow existing code style
5. Test on multiple devices/browsers
6. Update README for new features

## 📄 License

Same as parent project (AdWhisper).

## 🔗 Links

- [Backend README](/backend/README.md)
- [Architecture Documentation](/backend/AGENT_ARCHITECTURE.md)
- [API Documentation](/backend/README.md#api-endpoints)

---

**Built with ❤️ for fair and inclusive advertising**
