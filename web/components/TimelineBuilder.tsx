'use client';

import { useState, useEffect } from 'react';
import { PatientTimeline, Event, ExampleTimeline } from '@/types';
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
    // Load example timelines when task type changes
    loadExampleTimelines();
    setSelectedExample('');
    setCurrentTimeline(null);
  }, [taskType]);

  const loadExampleTimelines = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/example/${taskType}`);
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

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800">Patient Timeline</h2>
        {currentTimeline && (
          <div className="text-sm text-gray-600">
            <span className="font-medium">{currentTimeline.events.length}</span> events
          </div>
        )}
      </div>

      {/* Example Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Example Case
        </label>
        <select
          value={selectedExample}
          onChange={(e) => handleExampleSelect(e.target.value)}
          disabled={loading}
          className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
        >
          <option value="">Choose a patient example...</option>
          {exampleTimelines.map((example) => (
            <option key={example.name} value={example.name}>
              {example.name} - Expected Risk: {example.expected_risk}
            </option>
          ))}
        </select>
        {selectedExampleData && (
          <p className="mt-2 text-sm text-gray-600">
            {selectedExampleData.description}
          </p>
        )}
      </div>

      {/* Timeline Visualization */}
      {currentTimeline && (
        <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
          <TimelineVisualization 
            timeline={currentTimeline}
            highlightedEvents={[]}
          />
        </div>
      )}

      {/* Patient Info */}
      {currentTimeline && (
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded p-3">
            <p className="text-xs text-blue-600 font-medium">Subject ID</p>
            <p className="text-sm font-semibold text-gray-800">{currentTimeline.subject_id}</p>
          </div>
          {currentTimeline.stay_id && (
            <div className="bg-blue-50 rounded p-3">
              <p className="text-xs text-blue-600 font-medium">Stay ID</p>
              <p className="text-sm font-semibold text-gray-800">{currentTimeline.stay_id}</p>
            </div>
          )}
        </div>
      )}

      {/* Event Summary */}
      {currentTimeline && (
        <div className="mt-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Event Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {['vital', 'lab', 'medication', 'other'].map(eventType => {
              const count = currentTimeline.events.filter(e => e.type === eventType).length;
              return (
                <div key={eventType} className="bg-white border border-gray-200 rounded p-2">
                  <p className="text-xs text-gray-500 capitalize">{eventType}s</p>
                  <p className="text-lg font-semibold text-gray-800">{count}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {loading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-sm text-gray-600">Loading examples...</p>
        </div>
      )}
    </div>
  );
}
