// Type definitions matching API schemas

export interface Event {
  time: string;
  type: 'vital' | 'lab' | 'medication' | 'procedure' | 'diagnosis' | string;
  code: string | null;
  value: number | null;
  unit?: string;
}

export interface StaticFeatures {
  age?: number;
  sex?: 'M' | 'F' | 'O' | 'U';
  comorbidity_chf?: boolean;
  comorbidity_renal?: boolean;
  comorbidity_liver?: boolean;
  comorbidity_copd?: boolean;
  comorbidity_diabetes?: boolean;
}

export interface PatientTimeline {
  subject_id: string;
  stay_id: string | null;
  events: Event[];
  static_features?: StaticFeatures | Record<string, any>;
}

export interface ContributingEvent {
  time: string;
  type: string;
  code: string;
  value: number | null;
  contribution_score: number;
}

export interface PredictionResponse {
  task: 'readmission' | 'icu_mortality';
  risk_score: number;
  risk_label: 'low' | 'medium' | 'high';
  explanation: string;
  contributing_events: ContributingEvent[];
  model_name: string;
  model_version?: string;
}

export interface ExampleTimeline {
  name: string;
  description: string;
  timeline: PatientTimeline;
  expected_risk: 'Low' | 'Medium' | 'High';
}

export interface ModelInfo {
  task: 'readmission' | 'icu_mortality';
  model_name: string;
  model_type: string;
  version: string;
  metrics: Record<string, any>;
  trained_date?: string;
}
