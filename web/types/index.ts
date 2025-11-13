// Type definitions matching API schemas

export interface Event {
  time: string;
  type: 'vital' | 'lab' | 'medication' | 'admission' | 'discharge' | 'icu_in' | 'icu_out' | 'other';
  code: string | null;
  value: number | null;
  unit?: string;
}

export interface PatientTimeline {
  subject_id: string;
  stay_id: string | null;
  events: Event[];
  static_features?: Record<string, any>;
}

export interface ContributingEvent {
  feature_name: string;
  time_period: string;
  value: number | string;
  contribution: number;
  interpretation: string;
}

export interface PredictionResponse {
  task: string;
  risk_score: number;
  risk_label: 'Low' | 'Medium' | 'High';
  explanation: string;
  contributing_events: ContributingEvent[];
  model_type: string;
  prediction_time: string;
}

export interface ExampleTimeline {
  name: string;
  description: string;
  timeline: PatientTimeline;
  expected_risk: 'Low' | 'Medium' | 'High';
}
