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

  const eventTypeConfig: Record<string, { bg: string; border: string; text: string; icon: string }> = {
    vital: {
      bg: 'bg-emerald-50',
      border: 'border-emerald-400',
      text: 'text-emerald-700',
      icon: 'M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z',
    },
    lab: {
      bg: 'bg-blue-50',
      border: 'border-blue-400',
      text: 'text-blue-700',
      icon: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z',
    },
    medication: {
      bg: 'bg-purple-50',
      border: 'border-purple-400',
      text: 'text-purple-700',
      icon: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z',
    },
    procedure: {
      bg: 'bg-orange-50',
      border: 'border-orange-400',
      text: 'text-orange-700',
      icon: 'M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253',
    },
    diagnosis: {
      bg: 'bg-rose-50',
      border: 'border-rose-400',
      text: 'text-rose-700',
      icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z',
    },
  };

  const defaultConfig = {
    bg: 'bg-slate-50',
    border: 'border-slate-400',
    text: 'text-slate-700',
    icon: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  };

  const formatDateTime = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      return {
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        time: date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      };
    } catch {
      return { date: dateStr, time: '' };
    }
  };

  const formatEventValue = (event: Event) => {
    if (event.value !== null && event.value !== undefined) {
      return `${event.value}${event.unit ? ' ' + event.unit : ''}`;
    }
    return null;
  };

  if (sortedEvents.length === 0) {
    return (
      <div className="text-center py-12 text-slate-500">
        <svg className="w-12 h-12 mx-auto mb-3 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-sm">No events in timeline</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Legend */}
      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600 pb-3 border-b border-slate-200">
        {Object.entries(eventTypeConfig).slice(0, 4).map(([type, config]) => (
          <div key={type} className="flex items-center">
            <div className={`w-2.5 h-2.5 rounded-full ${config.border.replace('border-', 'bg-')} mr-1.5`}></div>
            <span className="capitalize">{type}s</span>
          </div>
        ))}
        {highlightedEvents.length > 0 && (
          <div className="flex items-center ml-auto">
            <div className="w-2.5 h-2.5 rounded-full bg-amber-400 ring-2 ring-amber-200 mr-1.5"></div>
            <span className="font-medium text-amber-700">Key Event</span>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-[22px] top-2 bottom-2 w-0.5 bg-gradient-to-b from-slate-200 via-slate-300 to-slate-200"></div>

        {/* Events */}
        <div className="space-y-3">
          {sortedEvents.map((event, index) => {
            const isHighlighted = highlightedEvents.includes(event.code || '');
            const config = eventTypeConfig[event.type] || defaultConfig;
            const { date, time } = formatDateTime(event.time);
            const value = formatEventValue(event);

            return (
              <div key={index} className="relative flex items-start group">
                {/* Timeline dot */}
                <div className={`relative z-10 flex-shrink-0 w-11 flex items-center justify-center`}>
                  <div className={`w-3.5 h-3.5 rounded-full border-2 transition-all ${
                    isHighlighted 
                      ? 'bg-amber-400 border-amber-500 ring-4 ring-amber-100 scale-125' 
                      : `bg-white ${config.border}`
                  }`}></div>
                </div>

                {/* Event card */}
                <div className={`flex-1 ml-2 p-3 rounded-xl border-l-4 transition-all ${
                  isHighlighted 
                    ? 'bg-amber-50 border-amber-400 ring-1 ring-amber-200 shadow-sm' 
                    : `${config.bg} ${config.border} group-hover:shadow-sm`
                }`}>
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center flex-wrap gap-2">
                        <span className={`text-xs font-semibold uppercase tracking-wide ${
                          isHighlighted ? 'text-amber-700' : config.text
                        }`}>
                          {event.type}
                        </span>
                        {event.code && (
                          <span className="text-xs font-mono bg-white/70 px-1.5 py-0.5 rounded text-slate-600">
                            {event.code}
                          </span>
                        )}
                        {isHighlighted && (
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-amber-500 text-white">
                            Key
                          </span>
                        )}
                      </div>
                      {value && (
                        <div className="mt-1.5 text-sm font-semibold text-slate-900">
                          {value}
                        </div>
                      )}
                    </div>
                    <div className="flex-shrink-0 text-right ml-3">
                      <div className="text-xs font-medium text-slate-700">{date}</div>
                      <div className="text-xs text-slate-500">{time}</div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Timeline Duration */}
      {sortedEvents.length > 1 && (
        <div className="pt-3 border-t border-slate-200">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>
              <span className="font-medium text-slate-700">{sortedEvents.length}</span> events
            </span>
            <span>
              {formatDateTime(sortedEvents[0].time).date} → {formatDateTime(sortedEvents[sortedEvents.length - 1].time).date}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
