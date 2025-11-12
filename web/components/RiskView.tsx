'use client';

import { PredictionResponse, PatientTimeline } from '@/types';
import TimelineVisualization from './TimelineVisualization';

interface RiskViewProps {
  prediction: PredictionResponse;
  timeline: PatientTimeline;
  taskType: 'readmission' | 'icu_mortality';
  modelType: 'logistic' | 'gru' | 'transformer';
}

export default function RiskView({ prediction, timeline, taskType, modelType }: RiskViewProps) {
  const riskPercentage = (prediction.risk_score * 100).toFixed(1);
  
  const riskLabelColors = {
    Low: 'bg-green-100 text-green-800 border-green-500',
    Medium: 'bg-yellow-100 text-yellow-800 border-yellow-500',
    High: 'bg-red-100 text-red-800 border-red-500',
  };

  const riskBarColors = {
    Low: 'bg-green-500',
    Medium: 'bg-yellow-500',
    High: 'bg-red-500',
  };

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
    .map(ce => {
      // Extract feature code from feature_name if present
      const match = ce.feature_name.match(/\(([^)]+)\)/);
      return match ? match[1] : null;
    })
    .filter((code): code is string => code !== null);

  return (
    <div className="space-y-6">
      {/* Risk Score Card */}
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 px-6 py-4">
          <h2 className="text-2xl font-bold text-white">Risk Assessment</h2>
          <p className="text-blue-100 text-sm mt-1">
            {taskDescriptions[taskType]} - {modelDescriptions[modelType]}
          </p>
        </div>

        <div className="p-6">
          {/* Risk Score Display */}
          <div className="text-center mb-6">
            <div className="inline-flex items-baseline">
              <span className="text-6xl font-bold text-gray-900">{riskPercentage}</span>
              <span className="text-3xl font-semibold text-gray-600 ml-2">%</span>
            </div>
            <div className={`inline-block mt-4 px-6 py-2 rounded-full text-lg font-semibold border-2 ${riskLabelColors[prediction.risk_label]}`}>
              {prediction.risk_label} Risk
            </div>
          </div>

          {/* Risk Bar */}
          <div className="mb-6">
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <div
                className={`h-full ${riskBarColors[prediction.risk_label]} transition-all duration-500`}
                style={{ width: `${riskPercentage}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>0%</span>
              <span>25%</span>
              <span>50%</span>
              <span>75%</span>
              <span>100%</span>
            </div>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Model:</span>
              <span className="ml-2 font-medium text-gray-900">{prediction.model_type}</span>
            </div>
            <div>
              <span className="text-gray-600">Prediction Time:</span>
              <span className="ml-2 font-medium text-gray-900">
                {new Date(prediction.prediction_time).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Clinical Explanation</h3>
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
          <p className="text-gray-700 leading-relaxed">{prediction.explanation}</p>
        </div>
      </div>

      {/* Contributing Events */}
      {prediction.contributing_events.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Key Contributing Factors</h3>
          <div className="space-y-3">
            {prediction.contributing_events.map((event, index) => (
              <div
                key={index}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{event.feature_name}</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      <span className="font-medium">Time Period:</span> {event.time_period}
                    </p>
                    <p className="text-sm text-gray-600">
                      <span className="font-medium">Value:</span> {event.value}
                    </p>
                    <p className="text-sm text-gray-700 mt-2">{event.interpretation}</p>
                  </div>
                  <div className="flex-shrink-0 ml-4">
                    <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      event.contribution > 0 
                        ? 'bg-red-100 text-red-800' 
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {event.contribution > 0 ? '+' : ''}{(event.contribution * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Timeline with Highlights */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Timeline with Key Events Highlighted
        </h3>
        <TimelineVisualization 
          timeline={timeline}
          highlightedEvents={highlightedEvents}
        />
      </div>

      {/* Disclaimer */}
      <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
        <h4 className="text-sm font-semibold text-red-800 mb-1">Research Prototype Disclaimer</h4>
        <p className="text-xs text-red-700">
          This prediction is generated by a research model on de-identified or synthetic data. 
          It has not been validated for clinical use and should not be used for patient diagnosis, 
          treatment planning, or any medical decision-making. This system is for educational and 
          research purposes only.
        </p>
      </div>
    </div>
  );
}
