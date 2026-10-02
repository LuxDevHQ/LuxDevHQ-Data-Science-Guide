# RentFlow – Car Rental Management System

**Python OOP Capstone Project**

---

## The Scenario

**SafariDrive Rentals** is a growing car rental company in Kenya. Today, it tracks its vehicles, customers, bookings, payments and returns by hand, in notebooks and spreadsheets.

As the fleet grows, staff struggle to answer simple questions:

- Which vehicles are available right now?
- Which vehicles are rented, and who has them?
- How long has each vehicle been out?
- How much does a customer owe, and have they paid?
- Was a vehicle returned late?
- How much money has the business made?

Your job is to build **RentFlow**, a Python program that answers all of these questions using **Object-Oriented Programming (OOP)**.

> **RentFlow** is the software you are building. **SafariDrive Rentals** is the company using it. You will see both names in this brief.

---

## What You Will Learn

By the end of this project you will have practised all four pillars of OOP in one working system:

| Pillar | Where you will use it |
|---|---|
| **Abstraction** | `Vehicle` and `Payment` define *what* every vehicle or payment must do, without saying *how*. |
| **Inheritance** | `EconomyCar`, `SUV` and `LuxuryCar` reuse everything in `Vehicle`. The three payment types reuse `Payment`. |
| **Polymorphism** | `calculate_rental_cost()` and `process_payment()` are called the same way but behave differently for each class. |
| **Encapsulation** | Vehicle availability, payment status and business revenue are protected and can only change through methods. |

You will also practise lists of objects, loops, conditionals, `try`/`except`, `super()`, and splitting a program across several files.

---

## Before You Start

You should be comfortable with:

- Writing classes, `__init__`, and instance methods
- Lists and dictionaries
- `for` loops and `if` statements
- Raising and catching exceptions (`raise ValueError(...)`, `try`/`except`)

You will meet these, and it is fine if they are new:

- `abc.ABC` and `@abstractmethod` (for abstract classes)
- Private attributes using double underscores (`self.__available`)
- `@property` (for read-only access to private data)

---

## Project Structure

Split your code into separate files. Do **not** put everything in `main.py`.

```text
rentflow/
│
├── main.py            # Runs the demo or the menu
├── vehicle.py         # Vehicle, EconomyCar, SUV, LuxuryCar
├── customer.py        # Customer
├── rental.py          # Rental
├── payment.py         # Payment, MpesaPayment, CardPayment, CashPayment
├── rental_system.py   # CarRentalSystem
│
└── README.md
```

---

## System Overview

### Class diagram

```text
                 Vehicle  (abstract)
                     |
      -------------------------------
      |              |              |
 EconomyCar         SUV         LuxuryCar


                 Payment  (abstract)
                     |
      -------------------------------
      |              |              |
 MpesaPayment   CardPayment    CashPayment


 Customer         Rental         CarRentalSystem
```

### How the classes work together

| Class | Job | Knows about |
|---|---|---|
| `Vehicle` (+ children) | Stores car details, tracks availability, calculates cost | Nothing else |
| `Customer` | Stores customer details and their rental history | Their `Rental` objects |
| `Rental` | Links one customer to one vehicle and tracks the bill | A `Customer` and a `Vehicle` |
| `Payment` (+ children) | Pays the bill for one rental | A `Rental` |
| `CarRentalSystem` | Runs the business: holds every list, enforces the rules, tracks revenue | Everything |

You may add extra classes if they make your design cleaner.

---

## Part 1 – The `Vehicle` Class (Abstract)

`Vehicle` is the parent of every car in the fleet. You should **never** create a plain `Vehicle` object, only an `EconomyCar`, `SUV` or `LuxuryCar`.

### Attributes

| Attribute | Example |
|---|---|
| `registration_number` | `KDK 123A` |
| `make` | `Toyota` |
| `model` | `Axio` |
| `year` | `2022` |
| `daily_rate` | `4000` |
| `__available` *(private)* | `True` |

### Methods

| Method | What it does |
|---|---|
| `display_details()` | Prints the vehicle's details and status |
| `is_available()` | Returns `True` or `False` |
| `mark_as_rented()` | Sets availability to `False` |
| `mark_as_available()` | Sets availability to `True` |
| `calculate_rental_cost(days)` | **Abstract.** Each child class must write its own version |

### Starter code

```python
from abc import ABC, abstractmethod


class Vehicle(ABC):
    def __init__(self, registration_number, make, model, year, daily_rate):
        if daily_rate < 0:
            raise ValueError("Daily rate cannot be negative.")

        self.registration_number = registration_number
        self.make = make
        self.model = model
        self.year = year
        self.daily_rate = daily_rate
        self.__available = True          # private

    def is_available(self):
        return self.__available

    def mark_as_rented(self):
        self.__available = False

    def mark_as_available(self):
        self.__available = True

    @abstractmethod
    def calculate_rental_cost(self, days):
        pass
```

Try this: `Vehicle("KDK 123A", "Toyota", "Axio", 2022, 4000)`. Python will refuse to create it. That refusal is **abstraction** at work.

---

## Part 2 – Vehicle Types (Inheritance and Polymorphism)

Create three child classes. Each one calls `super().__init__(...)` and then writes its **own** `calculate_rental_cost(days)`.

| Class | Example models | Pricing rule |
|---|---|---|
| `EconomyCar` | Toyota Axio, Toyota Vitz, Mazda Demio | `daily_rate × days` |
| `SUV` | Toyota Prado, Mazda CX-5, Nissan X-Trail | `daily_rate × days` **+ KSh 2,000** service charge (once per rental) |
| `LuxuryCar` | Mercedes-Benz E-Class, BMW 5 Series | `daily_rate × days` **+ 10%** insurance on that amount |

### Worked examples (3 days each)

```text
Economy – Toyota Axio @ KSh 4,000/day
  4,000 × 3                     = KSh 12,000

SUV – Toyota Prado @ KSh 8,000/day
  8,000 × 3                     = KSh 24,000
  Service charge                = KSh  2,000
  Total                         = KSh 26,000

Luxury – Mercedes E-Class @ KSh 15,000/day
  15,000 × 3                    = KSh 45,000
  Insurance (10% of 45,000)     = KSh  4,500
  Total                         = KSh 49,500
```

### Why this is polymorphism

```python
for car in [axio, prado, mercedes]:
    print(car.calculate_rental_cost(3))
```

The loop calls the **same method name** on every car and never checks what type it is. Each object still works out its own price.

---

## Part 3 – The `Customer` Class

### Attributes

| Attribute | Example |
|---|---|
| `customer_id` | `C001` |
| `name` | `John Kamau` |
| `phone_number` | `0712345678` |
| `national_id` | `12345678` |
| `driving_licence` | `DL45821` |
| `rentals` | `[]` (a list of this customer's `Rental` objects) |

### Methods

| Method | What it does |
|---|---|
| `display_details()` | Prints the customer's details |
| `add_rental(rental)` | Adds a rental to their history |
| `view_rental_history()` | Prints every rental this customer has made |

A customer **must** have a driving licence. If it is empty, raise a `ValueError`.

---

## Part 4 – The `Rental` Class

A rental connects one customer to one vehicle:

```text
Customer  +  Vehicle  =  Rental
```

### Attributes

| Attribute | Meaning |
|---|---|
| `rental_id` | e.g. `R001` |
| `customer` | The `Customer` object |
| `vehicle` | The `Vehicle` object |
| `rental_days` | Days the customer *agreed* to rent for |
| `actual_days` | Days the car was *actually* kept (set on return) |
| `status` | `ACTIVE`, `COMPLETED` or `CANCELLED` |
| `base_cost` | From `vehicle.calculate_rental_cost(rental_days)` |
| `late_fee` | Starts at `0` |
| `damage_charge` | Starts at `0` |
| `__paid` *(private)* | Starts as `False` |

### Methods

| Method | What it does |
|---|---|
| `calculate_late_fee()` | Works out the penalty for extra days |
| `calculate_total()` | Returns `base_cost + late_fee + damage_charge` |
| `complete_rental(actual_days, damage_charge=0)` | Records the return and sets status to `COMPLETED` |
| `mark_as_paid()` / `is_paid()` | Controls and reports payment status |
| `display_rental_details()` | Prints a receipt |

### The rental lifecycle

```text
          rent_vehicle()               return_vehicle()
 (none) ─────────────────► ACTIVE ─────────────────────► COMPLETED
                             │
                             │ cancel_rental()  (optional)
                             ▼
                         CANCELLED
```

### Late-return penalty

```text
Late days          = actual_days − rental_days   (0 if returned on time or early)
Daily late penalty = 20% of the vehicle's daily rate
Late fee           = daily late penalty × late days
```

**Example:** John rents the Toyota Prado (KSh 8,000/day) for 4 days but returns it after 6.

```text
Late days          = 6 − 4                = 2
Daily late penalty = 20% × 8,000          = KSh 1,600
Late fee           = 1,600 × 2            = KSh 3,200

Base cost (4 days + service charge)       = KSh 34,000
Late fee                                  = KSh  3,200
Final bill                                = KSh 37,200
```

The late fee is calculated on the daily rate only. It does **not** add a second service charge or insurance.

---

## Part 5 – The `Payment` Class (Abstract)

Every payment is linked to **one rental** and pays that rental's final bill.

```text
              Payment  (abstract)
                 |
      -----------------------
      |          |          |
   Mpesa       Card       Cash
```

### What every payment has

- `rental` – the rental being paid for
- `amount` – taken from `rental.calculate_total()`
- `process_payment()` – **abstract**, returns `True` if the payment succeeded and `False` if it failed

### How each type behaves differently

| Class | Extra information | Behaviour |
|---|---|---|
| `MpesaPayment` | Phone number | Generates a transaction code such as `TJK34H8ABC` and prints a confirmation |
| `CardPayment` | Last 4 digits of the card | Prints a confirmation like `Card ending 4417 charged KSh 26,000` |
| `CashPayment` | Amount handed over | Fails if the cash is less than the bill; otherwise prints the change due |

> **Tip:** Never store a full card number, even in a practice project. Storing only the last four digits is how real systems do it.

**Example output:**

```text
Processing M-Pesa payment...
Transaction: TJK34H8ABC
Payment of KSh 37,200 successful.
```

```text
Cash received: KSh 40,000
Change due:    KSh 2,800
Cash payment of KSh 37,200 recorded.
```

---

## Part 6 – The `CarRentalSystem` Class

This is the "manager" of the whole business.

### Attributes

```python
self.name = "SafariDrive Rentals"
self.vehicles = []
self.customers = []
self.rentals = []
self.payments = []
self.__revenue = 0        # private
```

### Methods

| Method | What it does |
|---|---|
| `add_vehicle(vehicle)` | Adds a vehicle to the fleet |
| `register_customer(customer)` | Adds a customer |
| `find_vehicle(registration_number)` | Returns the vehicle, or `None` |
| `find_customer(customer_id)` | Returns the customer, or `None` |
| `show_available_vehicles()` | Lists vehicles that can be rented |
| `search_vehicles(vehicle_type=None, max_daily_rate=None)` | Filters available vehicles |
| `rent_vehicle(customer, vehicle, days)` | Checks the rules, creates a `Rental`, marks the car rented |
| `return_vehicle(rental, actual_days, damage_charge=0)` | Completes the rental and frees the car |
| `process_payment(payment)` | Runs the payment and adds to revenue only if it succeeds |
| `show_active_rentals()` | Lists `ACTIVE` rentals |
| `show_completed_rentals()` | Lists `COMPLETED` rentals |
| `revenue` *(property)* | Returns total revenue (read-only) |
| `generate_report()` | Prints the management report |

### Searching for vehicles

```python
system.search_vehicles(vehicle_type="SUV")
```

```text
AVAILABLE SUVs

1. Toyota Prado
   Registration: KDJ 456B
   Rate: KSh 8,000/day

2. Nissan X-Trail
   Registration: KDJ 672B
   Rate: KSh 6,500/day
```

With a budget filter, only the X-Trail is returned:

```python
system.search_vehicles(vehicle_type="SUV", max_daily_rate=7000)
```

---

## Part 7 – Encapsulation in Practice

Some data is too important to be changed freely. Protect it with private attributes and change it only through methods.

| Protected data | Lives in | Can only change through |
|---|---|---|
| Availability | `Vehicle.__available` | `mark_as_rented()`, `mark_as_available()` |
| Payment status | `Rental.__paid` | `mark_as_paid()` |
| Revenue | `CarRentalSystem.__revenue` | `process_payment()` (after a successful payment) |

To let other code **read** revenue without changing it, use a property:

```python
@property
def revenue(self):
    return self.__revenue
```

Now `print(system.revenue)` works, but `system.revenue = 10_000_000` raises an error.

> **Good to know:** Writing `system.__revenue = 10_000_000` from *outside* the class does not crash, but it also does not touch the real revenue. Python quietly creates a new, unrelated attribute. The real value, stored inside the class, stays safe. Try it and print `system.revenue` to see for yourself.

---

## Part 8 – Business Rules

Your program must enforce these rules. When a rule is broken, **raise an exception** with a clear message, and **catch it** in `main.py` so the program does not crash.

| # | Rule | Suggested exception |
|---|---|---|
| 1 | Daily rates cannot be negative | `ValueError` |
| 2 | A customer must have a driving licence | `ValueError` |
| 3 | A customer must be registered before renting | `ValueError` |
| 4 | A vehicle must be in the fleet before it is rented | `ValueError` |
| 5 | Rental days must be greater than zero | `ValueError` |
| 6 | A rented vehicle cannot be rented again | `ValueError` |
| 7 | A vehicle cannot be returned unless it is currently rented | `ValueError` |
| 8 | A completed rental cannot be completed again | `ValueError` |
| 9 | Payment amounts cannot be negative | `ValueError` |
| 10 | A rental can only be paid once, and only after it is completed | `ValueError` |

You may create your own exception classes (for example `VehicleNotAvailableError`) as an extra challenge.

---

## Part 9 – The Complete Workflow

Your finished program must be able to run this whole journey:

```text
Register Customer
       ↓
Search for a Vehicle
       ↓
Rent an Available Vehicle  ──►  Vehicle becomes unavailable
       ↓
Return the Vehicle
       ↓
Add Late and Damage Charges
       ↓
Complete the Rental  ──────►  Vehicle becomes available again
       ↓
Process Payment  ──────────►  Revenue increases
       ↓
Generate Management Report
```

> Payment happens **after** the return, so the customer pays one final bill that already includes any late or damage charges.

---

## Part 10 – Sample `main.py`

```python
from vehicle import EconomyCar, SUV, LuxuryCar
from customer import Customer
from payment import MpesaPayment
from rental_system import CarRentalSystem

system = CarRentalSystem("SafariDrive Rentals")

# Add vehicles
car1 = EconomyCar("KDK 123A", "Toyota", "Axio", 2022, 4000)
car2 = SUV("KDJ 456B", "Toyota", "Prado", 2023, 8000)
car3 = SUV("KDJ 672B", "Nissan", "X-Trail", 2021, 6500)
car4 = LuxuryCar("KDL 789C", "Mercedes-Benz", "E-Class", 2024, 15000)

for car in [car1, car2, car3, car4]:
    system.add_vehicle(car)

# Register a customer
customer1 = Customer("C001", "John Kamau", "0712345678", "12345678", "DL45821")
system.register_customer(customer1)

# Rent the Prado for 4 days
system.show_available_vehicles()
rental1 = system.rent_vehicle(customer1, car2, 4)
rental1.display_rental_details()

# Try to rent it again (should fail)
try:
    system.rent_vehicle(customer1, car2, 2)
except ValueError as error:
    print(f"Error: {error}")

# Return it 2 days late, then pay
system.return_vehicle(rental1, actual_days=6)
payment1 = MpesaPayment(rental1, "0712345678")
system.process_payment(payment1)

system.generate_report()
```

---

## Part 11 – Expected Output

### Rental receipt

The extra-charge line changes with the vehicle type: a service charge for SUVs, insurance for luxury cars, and nothing for economy cars.

```text
====================================
        SAFARIDRIVE RENTALS
====================================
Rental ID:        R001
Customer:         John Kamau
Vehicle:          Toyota Prado
Registration:     KDJ 456B
Rental Period:    4 days
Daily Rate:       KSh 8,000

Rental Charge:    KSh 32,000
Service Charge:   KSh  2,000
------------------------------------
TOTAL:            KSh 34,000
Status:           ACTIVE
====================================
```

### Management report

```text
=====================================
    SAFARIDRIVE MANAGEMENT REPORT
=====================================
Total Vehicles:              15
Available Vehicles:           9
Rented Vehicles:              6

Registered Customers:        27

Total Rentals:               42
Active Rentals:               6
Completed Rentals:           36

Total Revenue:      KSh 785,000
-------------------------------------
VEHICLES BY CATEGORY
Economy Cars:                 7
SUVs:                         5
Luxury Cars:                  3
=====================================
```

Every number in the report must be **calculated from the objects in your lists**, never typed in by hand. Use `isinstance(vehicle, SUV)` to count each category.

---

## Suggested Build Order

Build and test one piece at a time. Do not move on until the current step works.

1. **Vehicles.** Write `Vehicle` and the three children. Create one of each and print their costs for 3 days.
2. **Customers.** Write `Customer` and test the driving-licence rule.
3. **Rentals.** Write `Rental` and test the late-fee example above (answer: KSh 3,200).
4. **The system.** Write `CarRentalSystem` with add, find, rent and return. Test that a rented car cannot be rented twice.
5. **Payments.** Write `Payment` and its children, then connect them to revenue.
6. **Report and receipt.** Add `generate_report()` and tidy your output.
7. **Bonus features**, if you have time.

---

## OOP Checklist

Tick each item before you submit.

- [ ] At least one abstract class (`Vehicle` and `Payment`)
- [ ] At least two parent classes
- [ ] At least five child classes
- [ ] `super().__init__()` used in every child class
- [ ] Methods overridden in child classes
- [ ] Polymorphism with `calculate_rental_cost()` and `process_payment()`
- [ ] Private attributes for availability, payment status and revenue
- [ ] Lists of objects
- [ ] Loops and conditionals
- [ ] `try` / `except` around actions that can fail
- [ ] Methods that return values
- [ ] Methods that change an object's state
- [ ] Code split across multiple files

---

## Bonus Features

Pick any of these once the main system works.

| Feature | Idea |
|---|---|
| **Automatic IDs** | Generate `C001, C002…` and `R001, R002…` with a class-level counter |
| **Dates** | Use `datetime` to set start and expected return dates, and work out `actual_days` from the real return date |
| **Maintenance status** | Add a `MAINTENANCE` state. A car under maintenance cannot be rented |
| **Damage charges** | On return, ask "Was the vehicle damaged?" and add a charge if yes |
| **Discounts** | Long-term rentals (e.g. 7+ days), returning customers, or corporate customers |
| **Cancellations** | `cancel_rental()` sets status to `CANCELLED` and frees the vehicle |
| **JSON storage** | Save and load customers, vehicles and rentals so data survives a restart |
| **SQLite database** | Store all data in an SQLite database instead of JSON |
| **Interactive menu** | Build a terminal menu (see below) |

```text
================================
     SAFARIDRIVE RENTALS
================================
1. Register Customer
2. Add Vehicle
3. View Available Vehicles
4. Rent Vehicle
5. Return Vehicle
6. Process Payment
7. View Active Rentals
8. View Completed Rentals
9. Generate Management Report
10. Exit

Choose an option:
```

---

## What to Submit

1. Complete Python source code
2. GitHub repository link
3. A `README.md` for your project
4. A class diagram (hand-drawn and photographed, or made with a tool such as draw.io)
5. Screenshots of the program running
6. Sample data: at least 5 vehicles, 3 customers and 3 rentals
7. A short explanation (a paragraph each) of how you used encapsulation, inheritance, polymorphism and abstraction
8. A short section on the real-world problem your system solves

# NOTE:
The project should be submitted to datascience@luxdevhq.com