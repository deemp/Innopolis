# Exercises
### Exercise 1
#### 1. Order countries by id asc, then show the 12th to 17th rows.

```sql
SELECT
  country,
  id
FROM
  countries
ORDER BY
  id ASC
LIMIT
  6 OFFSET 12;
```

#### 2. List all addresses in a city whose name starts with 'A’.


```sql
SELECT
  country,
  id
FROM
  countries
ORDER BY
  id ASC
LIMIT
  6 OFFSET 12;
```
