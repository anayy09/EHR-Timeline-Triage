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
    // Reset prediction when timeline changes
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
      const response = await fetch(`http://localhost:8000/api/predict/${selectedTask}?model_type=${selectedModel}`, {
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
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-blue-600 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">EHR Timeline Triage</h1>
          <p className="mt-2 text-blue-100">AI-powered readmission and mortality risk prediction</p>
        </div>
      </div>

      {/* Disclaimer Banner */}
      <div className="bg-red-50 border-l-4 border-red-600">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-600" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-800 font-semibold">
                RESEARCH PROTOTYPE ONLY - NOT FOR CLINICAL USE
              </p>
              <p className="mt-1 text-xs text-red-700">
                This system uses synthetic or de-identified data and has not been validated for clinical decision-making. 
                Do not use for patient diagnosis, treatment planning, or any medical purpose.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Task and Model Selection */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Task Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Prediction Task
              </label>
              <select
                value={selectedTask}
                onChange={(e) => setSelectedTask(e.target.value as 'readmission' | 'icu_mortality')}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="readmission">30-Day Readmission</option>
                <option value="icu_mortality">48-Hour ICU Mortality</option>
              </select>
              <p className="mt-1 text-xs text-gray-500">
                {selectedTask === 'readmission' 
                  ? 'Predicts risk of hospital readmission within 30 days of discharge'
                  : 'Predicts risk of death within 48 hours for ICU patients'}
              </p>
            </div>

            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Type
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as 'logistic' | 'gru' | 'transformer')}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="logistic">Logistic Regression</option>
                <option value="gru">GRU (Recurrent Neural Network)</option>
                <option value="transformer">Transformer</option>
              </select>
              <p className="mt-1 text-xs text-gray-500">
                {selectedModel === 'logistic' && 'Fast baseline with clear feature importance'}
                {selectedModel === 'gru' && 'Captures temporal patterns with recurrent architecture'}
                {selectedModel === 'transformer' && 'Advanced model with attention mechanisms'}
              </p>
            </div>
          </div>
        </div>

        {/* Timeline Builder */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <TimelineBuilder 
            onTimelineChange={handleTimelineChange} 
            taskType={selectedTask}
          />
        </div>

        {/* Predict Button */}
        {timeline && (
          <div className="flex justify-center mb-6">
            <button
              onClick={handlePredict}
              disabled={loading}
              className="px-8 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Computing Risk...
                </span>
              ) : (
                'Generate Risk Prediction'
              )}
            </button>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700 font-medium">Error: {error}</p>
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
          />
        )}
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-300 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="text-center">
            <p className="text-sm">
              EHR Timeline Triage v0.1.0 | Research Prototype
            </p>
            <p className="text-xs mt-2 text-gray-400">
              This system is for educational and research purposes only. Not FDA approved. Not for clinical use.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
