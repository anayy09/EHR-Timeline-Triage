'use client';

import { PatientTimeline, Event } from '@/types';
import { useMemo } from 'react';

interface TimelineVisualizationProps {
  timeline: PatientTimeline;
  highlightedEvents: string[];
}

export default function TimelineVisualization({ timeline, highlightedEvents }: TimelineVisualizationProps) {
  const sortedEvents = useMemo(() => {
    return [...timeline.events].sort((a, b) => 
      new Date(a.time).getTime() - new Date(b.time).getTime()
    );
  }, [timeline]);

  const eventTypeColors = {
    vital: 'bg-green-100 border-green-500 text-green-800',
    lab: 'bg-blue-100 border-blue-500 text-blue-800',
    medication: 'bg-purple-100 border-purple-500 text-purple-800',
    admission: 'bg-gray-100 border-gray-500 text-gray-800',
    discharge: 'bg-gray-100 border-gray-500 text-gray-800',
    icu_in: 'bg-red-100 border-red-500 text-red-800',
    icu_out: 'bg-red-100 border-red-500 text-red-800',
    other: 'bg-yellow-100 border-yellow-500 text-yellow-800',
  };

  const formatDateTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return {
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      time: date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    };
  };

  const formatEventValue = (event: Event) => {
    if (event.value !== null) {
      return `${event.value}${event.unit ? ' ' + event.unit : ''}`;
    }
    return null;
  };

  if (sortedEvents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No events in timeline
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-4 text-xs text-gray-600 mb-4">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
          <span>Vitals</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-500 rounded-full mr-1"></div>
          <span>Labs</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-purple-500 rounded-full mr-1"></div>
          <span>Medications</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
          <span>ICU</span>
        </div>
      </div>

      {/* Timeline */}
      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-300"></div>

        {/* Events */}
        <div className="space-y-4">
          {sortedEvents.map((event, index) => {
            const isHighlighted = highlightedEvents.includes(event.code || '');
            const colorClass = eventTypeColors[event.type] || eventTypeColors.other;
            const { date, time } = formatDateTime(event.time);
            const value = formatEventValue(event);

            return (
              <div key={index} className="relative flex items-start">
                {/* Timeline dot */}
                <div className={`absolute left-6 mt-1.5 w-4 h-4 rounded-full border-2 ${
                  isHighlighted 
                    ? 'bg-yellow-400 border-yellow-600 ring-4 ring-yellow-200' 
                    : 'bg-white border-gray-400'
                }`}></div>

                {/* Time label */}
                <div className="w-24 flex-shrink-0 text-right pr-4 pt-1">
                  <div className="text-xs font-medium text-gray-700">{date}</div>
                  <div className="text-xs text-gray-500">{time}</div>
                </div>

                {/* Event card */}
                <div className={`ml-8 flex-1 border-l-4 rounded-r-lg p-3 ${colorClass} ${
                  isHighlighted ? 'ring-2 ring-yellow-400' : ''
                }`}>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-semibold uppercase tracking-wide">
                          {event.type}
                        </span>
                        {event.code && (
                          <span className="text-xs font-mono bg-white bg-opacity-50 px-1 rounded">
                            {event.code}
                          </span>
                        )}
                      </div>
                      {value && (
                        <div className="mt-1 text-sm font-bold">
                          {value}
                        </div>
                      )}
                    </div>
                    {isHighlighted && (
                      <div className="flex-shrink-0">
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-600 text-white">
                          Key Event
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Summary */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="text-xs text-gray-600">
          <span className="font-medium">Timeline Duration:</span>{' '}
          {sortedEvents.length > 0 && (
            <>
              {formatDateTime(sortedEvents[0].time).date}{' '}
              {formatDateTime(sortedEvents[0].time).time} to{' '}
              {formatDateTime(sortedEvents[sortedEvents.length - 1].time).date}{' '}
              {formatDateTime(sortedEvents[sortedEvents.length - 1].time).time}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
