# EHR Timeline Triage - Web Frontend

Modern web dashboard for the EHR Timeline Triage risk prediction system.

> **⚠️ RESEARCH PROTOTYPE - NOT FOR CLINICAL USE**

## Overview

This is the frontend application for EHR Timeline Triage, providing an interactive interface for:
- Building patient timelines with clinical events
- Selecting prediction tasks and ML models
- Visualizing risk scores with explanations
- Exploring contributing factors and event attributions

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Next.js | 16.0.1 | React framework with App Router |
| React | 19.2.0 | UI library |
| React DOM | 19.2.0 | React DOM bindings |
| TypeScript | 5 | Type-safe JavaScript |
| Tailwind CSS | 4 | Utility-first CSS framework |
| PostCSS | latest | CSS processing |

## Prerequisites

- Node.js 20+ (LTS recommended)
- npm 10+ or yarn
- Backend API running at http://localhost:8000

## Getting Started

### Development Mode

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

### Using Docker

The frontend is included in the main `docker-compose.yml`. From the project root:

```bash
# Build and run all services
docker-compose up --build

# Or run just the web service
docker-compose up web
```

## Project Structure

```
web/
├── app/                    # Next.js App Router
│   ├── globals.css         # Global styles and Tailwind
│   ├── layout.tsx          # Root layout component
│   └── page.tsx            # Main page component
├── components/             # React components
│   ├── TimelineBuilder.tsx # Event input interface
│   ├── TimelineVisualization.tsx # Timeline display
│   └── RiskView.tsx        # Risk score display
├── types/                  # TypeScript definitions
│   └── index.ts            # Shared type interfaces
├── public/                 # Static assets
├── next.config.ts          # Next.js configuration
├── tailwind.config.ts      # Tailwind CSS configuration
├── tsconfig.json           # TypeScript configuration
├── package.json            # Dependencies and scripts
└── Dockerfile              # Container build file
```

## Components

### TimelineBuilder
Interactive form for adding clinical events:
- Vitals (Heart Rate, Blood Pressure, SpO2, etc.)
- Labs (Lactate, Creatinine, WBC, etc.)
- Medications (Vasopressors, Antibiotics, Sedatives)
- Static features (Age, Sex, Comorbidities)

### TimelineVisualization
Visual timeline display with:
- Chronological event ordering
- Event type color coding
- Highlighted contributing factors
- Interactive event details

### RiskView
Risk assessment display featuring:
- Circular risk score gauge
- Color-coded risk levels (Low/Medium/High)
- Natural language explanation
- Contributing factor breakdown
- Research disclaimer

## API Integration

The frontend communicates with the FastAPI backend:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/predict/{task}` | POST | Get risk prediction |
| `/api/example/{task}` | GET | Load example timelines |
| `/api/models` | GET | List available models |

### Environment Configuration

Create `.env.local` for development:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For Docker, the API URL is configured via environment variables in `docker-compose.yml`.

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server with hot reload |
| `npm run build` | Build production bundle |
| `npm start` | Start production server |
| `npm run lint` | Run ESLint |

## Styling

The application uses Tailwind CSS 4 with:
- Custom gradient backgrounds
- Responsive design
- Dark mode support (planned)
- Component-based styling

## Type Safety

TypeScript interfaces are defined in `types/index.ts`:

```typescript
interface PatientTimeline {
  subject_id: string;
  stay_id?: string;
  events: Event[];
  static_features?: StaticFeatures;
}

interface PredictionResponse {
  task: string;
  risk_score: number;
  risk_label: 'low' | 'medium' | 'high';
  explanation: string;
  contributing_events: ContributingEvent[];
  model_name: string;
  model_version: string;
}
```

## Features

### Prediction Tasks
- **30-Day Readmission**: Hospital readmission risk prediction
- **48-Hour ICU Mortality**: ICU mortality risk prediction

### Model Selection
- **Logistic Regression**: Fast baseline with interpretable coefficients
- **GRU**: Recurrent neural network for temporal patterns
- **Transformer**: Attention-based sequence model

### Risk Visualization
- Circular gauge with percentage display
- Three-tier risk classification
- Color-coded alerts (green/amber/red)

## Development

### Hot Reload
The development server supports hot module replacement. Changes to components will automatically reflect in the browser.

### Linting
```bash
npm run lint
```

### Type Checking
TypeScript errors are checked during build:
```bash
npm run build
```

## Deployment

### Vercel (Recommended for Next.js)
```bash
npx vercel
```

### Docker
Included Dockerfile supports production deployment:
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

## Troubleshooting

### API Connection Issues
1. Ensure backend is running at http://localhost:8000
2. Check CORS settings in backend
3. Verify `NEXT_PUBLIC_API_URL` environment variable

### Build Errors
1. Clear `.next` directory: `rm -rf .next`
2. Remove `node_modules`: `rm -rf node_modules`
3. Reinstall: `npm install`

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

## Related

- [Main Project README](../README.md)
- [API Documentation](http://localhost:8000/docs)
- [System Design](../docs/system_design.md)

---

## ⚠️ Disclaimer

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

This is a research prototype. It has NOT been validated for clinical use and should NOT be used for patient care decisions.
