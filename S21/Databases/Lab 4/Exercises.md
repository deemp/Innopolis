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
  DISTINCT customer
FROM
  customers
WHERE
  payment >= 11;
```

#### 4. List all customers' first name, last name and the city they live in.

```sql
SELECT
  first_name, last_name, city
FROM
  customers;
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

#### 1. Create 2 Views of your choice.

##### a. 

```sql
CREATE VIEW cat AS SELECT * FROM category;
```

##### b.

```sql
CREATE VIEW f AS SELECT * FROM film;

```
#### 2. Implement one of the views into a query.

```sql
SELECT * INTO p FROM cat;
```


#### 3. Create a Trigger of your choice.
```sql
CREATE OR REPLACE FUNCTION rent_func()
 RETURNS TRIGGER 
  LANGUAGE PLPGSQL
  AS                          
$$
BEGIN
IF rental.rental_date < rental.return_date
THEN 
INSERT INTO country (country_id, country, last_update) 
VALUES (city.country_id, 'Gvatemala', now());
END IF;
RETURN rental;
END;
$$
;

CREATE TRIGGER rent_func_trigger
BEFORE UPDATE ON rental
FOR EACH ROW
EXECUTE PROCEDURE rent_func();
```
