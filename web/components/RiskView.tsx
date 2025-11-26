'use client';

import { PredictionResponse, PatientTimeline } from '@/types';
import TimelineVisualization from './TimelineVisualization';

interface RiskViewProps {
  prediction: PredictionResponse;
  timeline: PatientTimeline;
  taskType: 'readmission' | 'icu_mortality';
  modelType: 'logistic' | 'gru' | 'transformer';
  onReset?: () => void;
}

export default function RiskView({ prediction, timeline, taskType, modelType, onReset }: RiskViewProps) {
  const riskPercentage = (prediction.risk_score * 100).toFixed(1);
  const riskLabel = prediction.risk_label.charAt(0).toUpperCase() + prediction.risk_label.slice(1);
  
  const riskStyles = {
    low: {
      gradient: 'from-emerald-500 to-teal-600',
      bg: 'bg-emerald-50',
      border: 'border-emerald-200',
      text: 'text-emerald-700',
      badge: 'bg-emerald-100 text-emerald-800 border-emerald-300',
      ring: 'ring-emerald-500',
    },
    medium: {
      gradient: 'from-amber-500 to-orange-600',
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      text: 'text-amber-700',
      badge: 'bg-amber-100 text-amber-800 border-amber-300',
      ring: 'ring-amber-500',
    },
    high: {
      gradient: 'from-rose-500 to-red-600',
      bg: 'bg-rose-50',
      border: 'border-rose-200',
      text: 'text-rose-700',
      badge: 'bg-rose-100 text-rose-800 border-rose-300',
      ring: 'ring-rose-500',
    },
  };

  const style = riskStyles[prediction.risk_label] || riskStyles.medium;

  const taskDescriptions = {
    readmission: '30-Day Hospital Readmission',
    icu_mortality: '48-Hour ICU Mortality',
  };

  const modelDescriptions = {
    logistic: 'Logistic Regression',
    gru: 'GRU Neural Network',
    transformer: 'Transformer Model',
  };

  // Extract highlighted event codes from contributing events
  const highlightedEvents = prediction.contributing_events
    .map(ce => ce.code)
    .filter((code): code is string => code !== null);

  return (
    <div className="space-y-6">
      {/* Risk Score Card */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className={`bg-gradient-to-r ${style.gradient} px-6 py-5`}>
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">Risk Assessment</h2>
              <p className="text-white/80 text-sm mt-1">
                {taskDescriptions[taskType]} • {modelDescriptions[modelType]}
              </p>
            </div>
            {onReset && (
              <button
                onClick={onReset}
                className="px-4 py-2 bg-white/20 hover:bg-white/30 text-white rounded-lg text-sm font-medium transition-colors backdrop-blur-sm"
              >
                New Analysis
              </button>
            )}
          </div>
        </div>

        <div className="p-6">
          {/* Risk Score Display */}
          <div className="flex items-center justify-center mb-8">
            <div className="relative">
              {/* Circular Progress */}
              <svg className="w-40 h-40 transform -rotate-90">
                <circle
                  className="text-slate-100"
                  strokeWidth="12"
                  stroke="currentColor"
                  fill="transparent"
                  r="58"
                  cx="80"
                  cy="80"
                />
                <circle
                  className={style.text}
                  strokeWidth="12"
                  strokeDasharray={`${prediction.risk_score * 364} 364`}
                  strokeLinecap="round"
                  stroke="currentColor"
                  fill="transparent"
                  r="58"
                  cx="80"
                  cy="80"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-4xl font-bold text-slate-900">{riskPercentage}</span>
                <span className="text-sm text-slate-500">percent</span>
              </div>
            </div>
          </div>

          {/* Risk Label */}
          <div className="text-center mb-6">
            <span className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold border ${style.badge}`}>
              <span className={`w-2 h-2 rounded-full mr-2 bg-current`}></span>
              {riskLabel} Risk
            </span>
          </div>

          {/* Risk Bar */}
          <div className="mb-6">
            <div className="relative h-3 bg-gradient-to-r from-emerald-200 via-amber-200 to-rose-200 rounded-full overflow-hidden">
              <div 
                className="absolute top-0 bottom-0 w-1 bg-slate-900 rounded-full shadow-lg"
                style={{ left: `calc(${riskPercentage}% - 2px)` }}
              />
            </div>
            <div className="flex justify-between text-xs text-slate-500 mt-2">
              <span>Low Risk</span>
              <span>Medium Risk</span>
              <span>High Risk</span>
            </div>
          </div>

          {/* Model Info */}
          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-slate-100">
            <div className="text-center">
              <p className="text-xs text-slate-500 uppercase tracking-wide">Model</p>
              <p className="text-sm font-medium text-slate-900 mt-1">{prediction.model_name}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-slate-500 uppercase tracking-wide">Version</p>
              <p className="text-sm font-medium text-slate-900 mt-1">{prediction.model_version || '0.1.0'}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-slate-50 to-white border-b border-slate-100">
          <h3 className="text-lg font-semibold text-slate-900 flex items-center">
            <svg className="w-5 h-5 mr-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Clinical Explanation
          </h3>
        </div>
        <div className="p-6">
          <div className={`${style.bg} ${style.border} border rounded-xl p-4`}>
            <p className="text-slate-700 leading-relaxed whitespace-pre-line">{prediction.explanation}</p>
          </div>
        </div>
      </div>

      {/* Contributing Factors */}
      {prediction.contributing_events.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
          <div className="px-6 py-4 bg-gradient-to-r from-slate-50 to-white border-b border-slate-100">
            <h3 className="text-lg font-semibold text-slate-900 flex items-center">
              <svg className="w-5 h-5 mr-2 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Key Contributing Factors
            </h3>
          </div>
          <div className="p-6">
            <div className="space-y-3">
              {prediction.contributing_events.slice(0, 5).map((event, index) => (
                <div
                  key={index}
                  className="flex items-center p-4 rounded-xl bg-slate-50 border border-slate-100 hover:bg-slate-100 transition-colors"
                >
                  <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-white border border-slate-200 flex items-center justify-center mr-4">
                    <span className="text-sm font-bold text-slate-600">#{index + 1}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center">
                      <span className="text-sm font-medium text-slate-900">{event.code}</span>
                      <span className="ml-2 px-2 py-0.5 text-xs rounded bg-slate-200 text-slate-600 uppercase">
                        {event.type}
                      </span>
                    </div>
                    <div className="flex items-center mt-1 text-xs text-slate-500">
                      <span>{event.time}</span>
                      {event.value !== null && (
                        <>
                          <span className="mx-2">•</span>
                          <span>Value: {event.value}</span>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex-shrink-0 ml-4">
                    <div className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      event.contribution_score > 0 
                        ? 'bg-rose-100 text-rose-700' 
                        : 'bg-emerald-100 text-emerald-700'
                    }`}>
                      {event.contribution_score > 0 ? '↑' : '↓'} {Math.abs(event.contribution_score).toFixed(3)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Timeline with Highlights */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-slate-50 to-white border-b border-slate-100">
          <h3 className="text-lg font-semibold text-slate-900 flex items-center">
            <svg className="w-5 h-5 mr-2 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Timeline with Key Events
          </h3>
        </div>
        <div className="p-6">
          <TimelineVisualization 
            timeline={timeline}
            highlightedEvents={highlightedEvents}
          />
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-gradient-to-r from-rose-50 to-orange-50 rounded-2xl border border-rose-200 p-6">
        <div className="flex items-start">
          <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-rose-100 flex items-center justify-center">
            <svg className="h-5 w-5 text-rose-600" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-4">
            <h4 className="text-sm font-semibold text-rose-900">Research Prototype Disclaimer</h4>
            <p className="mt-1 text-sm text-rose-700">
              This prediction is generated by a research model on de-identified or synthetic data. 
              It has not been validated for clinical use and should not be used for patient diagnosis, 
              treatment planning, or any medical decision-making.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
