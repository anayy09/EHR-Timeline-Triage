'use client';

import { useState } from 'react';
import TimelineBuilder from '@/components/TimelineBuilder';
import RiskView from '@/components/RiskView';
import { PatientTimeline, PredictionResponse } from '@/types';

export default function Home() {
  const [timeline, setTimeline] = useState<PatientTimeline | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [selectedTask, setSelectedTask] = useState<'readmission' | 'icu_mortality'>('readmission');
  const [selectedModel, setSelectedModel] = useState<'logistic' | 'gru' | 'transformer'>('logistic');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTimelineChange = (newTimeline: PatientTimeline) => {
    setTimeline(newTimeline);
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!timeline) {
      setError('Please create or select a patient timeline first.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/predict/${selectedTask}?model_type=${selectedModel}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(timeline),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const result: PredictionResponse = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      {/* Modern Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900 tracking-tight">EHR Timeline Triage</h1>
                <p className="text-xs text-slate-500">AI-Powered Clinical Risk Assessment</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800 border border-amber-200">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                Research Only
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Disclaimer Banner */}
      <div className="bg-gradient-to-r from-rose-50 to-orange-50 border-b border-rose-200">
        <div className="max-w-7xl mx-auto px-6 py-3">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-rose-100 flex items-center justify-center">
              <svg className="h-4 w-4 text-rose-600" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-rose-900">Research Prototype — Not for Clinical Use</p>
              <p className="text-xs text-rose-700 mt-0.5">
                This system uses synthetic data and is not validated for medical decision-making.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Configuration Panel */}
        <div className="mb-8">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="px-6 py-4 bg-gradient-to-r from-slate-50 to-white border-b border-slate-100">
              <h2 className="text-lg font-semibold text-slate-900 flex items-center">
                <svg className="w-5 h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Configuration
              </h2>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Task Selection */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-3">
                    Prediction Task
                  </label>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => setSelectedTask('readmission')}
                      className={`relative flex flex-col items-center p-4 rounded-xl border-2 transition-all ${
                        selectedTask === 'readmission'
                          ? 'border-indigo-500 bg-indigo-50 shadow-sm shadow-indigo-500/10'
                          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                      }`}
                    >
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center mb-2 ${
                        selectedTask === 'readmission' ? 'bg-indigo-500 text-white' : 'bg-slate-100 text-slate-600'
                      }`}>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                        </svg>
                      </div>
                      <span className={`text-sm font-medium ${
                        selectedTask === 'readmission' ? 'text-indigo-900' : 'text-slate-700'
                      }`}>30-Day Readmission</span>
                      <span className="text-xs text-slate-500 mt-1">Hospital return risk</span>
                    </button>
                    <button
                      onClick={() => setSelectedTask('icu_mortality')}
                      className={`relative flex flex-col items-center p-4 rounded-xl border-2 transition-all ${
                        selectedTask === 'icu_mortality'
                          ? 'border-indigo-500 bg-indigo-50 shadow-sm shadow-indigo-500/10'
                          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                      }`}
                    >
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center mb-2 ${
                        selectedTask === 'icu_mortality' ? 'bg-indigo-500 text-white' : 'bg-slate-100 text-slate-600'
                      }`}>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                      </div>
                      <span className={`text-sm font-medium ${
                        selectedTask === 'icu_mortality' ? 'text-indigo-900' : 'text-slate-700'
                      }`}>48-Hour ICU Mortality</span>
                      <span className="text-xs text-slate-500 mt-1">Critical care risk</span>
                    </button>
                  </div>
                </div>

                {/* Model Selection */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-3">
                    Model Architecture
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { value: 'logistic', label: 'Logistic', desc: 'Fast & interpretable' },
                      { value: 'gru', label: 'GRU', desc: 'Temporal patterns' },
                      { value: 'transformer', label: 'Transformer', desc: 'Attention-based' },
                    ].map((model) => (
                      <button
                        key={model.value}
                        onClick={() => setSelectedModel(model.value as any)}
                        className={`p-3 rounded-xl border-2 transition-all text-center ${
                          selectedModel === model.value
                            ? 'border-indigo-500 bg-indigo-50'
                            : 'border-slate-200 bg-white hover:border-slate-300'
                        }`}
                      >
                        <span className={`block text-sm font-medium ${
                          selectedModel === model.value ? 'text-indigo-900' : 'text-slate-700'
                        }`}>{model.label}</span>
                        <span className="text-xs text-slate-500">{model.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Timeline Builder */}
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="px-6 py-4 bg-gradient-to-r from-slate-50 to-white border-b border-slate-100">
              <h2 className="text-lg font-semibold text-slate-900 flex items-center">
                <svg className="w-5 h-5 mr-2 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Patient Timeline
              </h2>
            </div>
            <div className="p-6">
              <TimelineBuilder 
                onTimelineChange={handleTimelineChange} 
                taskType={selectedTask}
              />
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Predict Button */}
            {timeline && !prediction && (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-indigo-100 to-purple-100 flex items-center justify-center">
                    <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 mb-2">Ready to Analyze</h3>
                  <p className="text-sm text-slate-600 mb-6">
                    Timeline loaded with {timeline.events.length} events. Click below to generate risk assessment.
                  </p>
                  <button
                    onClick={handlePredict}
                    disabled={loading}
                    className="inline-flex items-center px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold rounded-xl shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    {loading ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        Generate Risk Assessment
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="bg-rose-50 rounded-2xl border border-rose-200 p-6">
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-rose-100 flex items-center justify-center">
                    <svg className="h-5 w-5 text-rose-600" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-4">
                    <h3 className="text-sm font-semibold text-rose-900">Prediction Error</h3>
                    <p className="mt-1 text-sm text-rose-700">{error}</p>
                    <button 
                      onClick={() => setError(null)}
                      className="mt-3 text-sm font-medium text-rose-600 hover:text-rose-800"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Risk View */}
            {prediction && (
              <RiskView 
                prediction={prediction} 
                timeline={timeline!}
                taskType={selectedTask}
                modelType={selectedModel}
                onReset={() => {
                  setPrediction(null);
                }}
              />
            )}

            {/* Empty State */}
            {!timeline && !prediction && (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center">
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-slate-100 to-slate-50 flex items-center justify-center">
                  <svg className="w-10 h-10 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-slate-900 mb-2">No Timeline Selected</h3>
                <p className="text-sm text-slate-600">
                  Select an example patient case from the timeline builder to get started.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 bg-slate-900 text-slate-400">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <span className="text-sm text-slate-300">EHR Timeline Triage v0.1.0</span>
            </div>
            <p className="text-xs text-center md:text-right">
              Research prototype for educational purposes only. Not FDA approved. Not for clinical use.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
