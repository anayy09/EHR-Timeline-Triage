'use client';

import { useState, useEffect } from 'react';
import { PatientTimeline, ExampleTimeline } from '@/types';
import TimelineVisualization from './TimelineVisualization';

interface TimelineBuilderProps {
  onTimelineChange: (timeline: PatientTimeline) => void;
  taskType: 'readmission' | 'icu_mortality';
}

export default function TimelineBuilder({ onTimelineChange, taskType }: TimelineBuilderProps) {
  const [exampleTimelines, setExampleTimelines] = useState<ExampleTimeline[]>([]);
  const [selectedExample, setSelectedExample] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [currentTimeline, setCurrentTimeline] = useState<PatientTimeline | null>(null);

  useEffect(() => {
    loadExampleTimelines();
    setSelectedExample('');
    setCurrentTimeline(null);
  }, [taskType]);

  const loadExampleTimelines = async () => {
    setLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/example/${taskType}`);
      if (response.ok) {
        const data = await response.json();
        setExampleTimelines(data.examples || []);
      }
    } catch (err) {
      console.error('Failed to load examples:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleSelect = (exampleName: string) => {
    setSelectedExample(exampleName);
    const example = exampleTimelines.find(e => e.name === exampleName);
    if (example) {
      setCurrentTimeline(example.timeline);
      onTimelineChange(example.timeline);
    }
  };

  const selectedExampleData = exampleTimelines.find(e => e.name === selectedExample);

  const getRiskBadgeStyle = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low':
        return 'bg-emerald-100 text-emerald-700 border-emerald-200';
      case 'high':
        return 'bg-rose-100 text-rose-700 border-rose-200';
      default:
        return 'bg-amber-100 text-amber-700 border-amber-200';
    }
  };

  return (
    <div>
      {/* Example Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-700 mb-3">
          Select Example Case
        </label>
        
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-indigo-500 border-t-transparent"></div>
            <span className="ml-3 text-sm text-slate-600">Loading examples...</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3">
            {exampleTimelines.map((example) => (
              <button
                key={example.name}
                onClick={() => handleExampleSelect(example.name)}
                className={`relative flex items-start p-4 rounded-xl border-2 text-left transition-all ${
                  selectedExample === example.name
                    ? 'border-indigo-500 bg-indigo-50 shadow-sm shadow-indigo-500/10'
                    : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                }`}
              >
                <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center mr-4 ${
                  selectedExample === example.name ? 'bg-indigo-500 text-white' : 'bg-slate-100 text-slate-600'
                }`}>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className={`font-medium ${
                      selectedExample === example.name ? 'text-indigo-900' : 'text-slate-900'
                    }`}>{example.name}</span>
                    <span className={`ml-2 px-2 py-0.5 text-xs font-medium rounded-full border ${getRiskBadgeStyle(example.expected_risk)}`}>
                      {example.expected_risk} Risk
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-slate-600">{example.description}</p>
                </div>
                {selectedExample === example.name && (
                  <div className="absolute top-4 right-4">
                    <svg className="w-5 h-5 text-indigo-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Timeline Visualization */}
      {currentTimeline && (
        <>
          {/* Patient Info Cards */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-4 border border-indigo-100">
              <div className="flex items-center">
                <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center mr-3">
                  <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V8a2 2 0 00-2-2h-5m-4 0V5a2 2 0 114 0v1m-4 0a2 2 0 104 0m-5 8a2 2 0 100-4 2 2 0 000 4zm0 0c1.306 0 2.417.835 2.83 2M9 14a3.001 3.001 0 00-2.83 2M15 11h3m-3 4h2" />
                  </svg>
                </div>
                <div>
                  <p className="text-xs text-indigo-600 font-medium">Subject ID</p>
                  <p className="text-sm font-semibold text-slate-900">{currentTimeline.subject_id}</p>
                </div>
              </div>
            </div>
            {currentTimeline.stay_id && (
              <div className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-xl p-4 border border-emerald-100">
                <div className="flex items-center">
                  <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center mr-3">
                    <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-xs text-emerald-600 font-medium">Stay ID</p>
                    <p className="text-sm font-semibold text-slate-900">{currentTimeline.stay_id}</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Event Summary */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-slate-700 mb-2">Event Summary</h3>
            <div className="grid grid-cols-4 gap-2">
              {[
                { type: 'vital', label: 'Vitals', color: 'emerald', icon: 'M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z' },
                { type: 'lab', label: 'Labs', color: 'blue', icon: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z' },
                { type: 'medication', label: 'Meds', color: 'purple', icon: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z' },
                { type: 'other', label: 'Other', color: 'slate', icon: 'M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z' },
              ].map(item => {
                const count = currentTimeline.events.filter(e => e.type === item.type).length;
                return (
                  <div key={item.type} className="bg-slate-50 rounded-lg p-3 text-center border border-slate-100">
                    <div className="text-2xl font-bold text-slate-900">{count}</div>
                    <div className="text-xs text-slate-500">{item.label}</div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Timeline */}
          <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
            <TimelineVisualization 
              timeline={currentTimeline}
              highlightedEvents={[]}
            />
          </div>
        </>
      )}

      {/* Empty State */}
      {!currentTimeline && !loading && exampleTimelines.length > 0 && (
        <div className="text-center py-8 text-slate-500">
          <svg className="w-12 h-12 mx-auto mb-3 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-sm">Select a patient example above to view their timeline</p>
        </div>
      )}
    </div>
  );
}
