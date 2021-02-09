# Exercises
### Exercise 1
#### 1. Order countries by id asc, then show the 12th to 17th rows.

```sql
SELECT
  country
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
  address
FROM
  addresses
WHERE
  LEFT (city, 1) = 'A';
```


#### 3. Find all customers with at least one payment whose amount is greater
than 11 dollars.


```sql
SELECT
  first_name, last_name, city
FROM
  customers;
```

#### 4. List all customers' first name, last name and the city they live in.

```sql
SELECT
  DISTINCT customer
FROM
  customers
WHERE
  payment >= 11;
```


#### 5. Find all duplicated first names in the customer table.

```sql
SELECT
  first_name,
  COUNT (first_name)
FROM
  customers
GROUP BY
  first_name
HAVING
  COUNT (first_name) > 1
```

### Exercise 2
