-- Additional Unity Catalog privileges required because this MLflow experiment
-- stores traces in UC tables. The App resource grants CAN_EDIT on the experiment,
-- while these grants allow its dedicated service principal to persist spans and
-- annotations. Replace the principal UUID if the App is deleted and recreated.

GRANT SELECT, MODIFY
ON TABLE workspace.car_manual_rag.`3683662210960230_otel_spans`
TO `62ac82bc-fb6f-4454-a75f-b424c8d5d067`;

GRANT SELECT, MODIFY
ON TABLE workspace.car_manual_rag.`3683662210960230_otel_annotations`
TO `62ac82bc-fb6f-4454-a75f-b424c8d5d067`;
