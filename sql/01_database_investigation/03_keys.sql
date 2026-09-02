SELECT tc.table_name, tc.constraint_type,
       kcu.column_name, tc.constraint_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
 AND tc.table_name = kcu.table_name
WHERE tc.table_schema = 'public'
  AND tc.constraint_type IN ('PRIMARY KEY','FOREIGN KEY')
ORDER BY tc.table_name, tc.constraint_type, kcu.ordinal_position;
