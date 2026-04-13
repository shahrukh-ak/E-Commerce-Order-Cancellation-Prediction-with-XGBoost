# E-Commerce Order Cancellation Prediction with XGBoost

Predicts whether a placed order will be cancelled using XGBoost with k-fold cross-validation. The project includes class balancing via under-sampling, datetime feature engineering, and one-hot encoding of high-cardinality categorical variables.

## Business Context

For an e-commerce wholesaler, a reliable cancellation signal at the point of order entry allows the operations team to prioritise manual review on high-risk orders rather than reviewing every order, reducing both workload and fulfilment costs.

## Dataset

`OnlineRetail-clean.csv` contains order-level transaction data including InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, ImputedCustomerID, and Country.

Cancellations are identified by an InvoiceNo that begins with `"C"`.

## Methodology

**Label Creation:** `IsCancellation` is derived from the InvoiceNo prefix.

**Class Balancing:** The majority class (non-cancelled orders) is under-sampled to match the number of cancellations. The combined dataset is further reduced by 50% to stay within a 10,000-row memory constraint.

**Feature Selection:** InvoiceNo, Description, Quantity, and ImputedCustomerID are dropped as they are either direct indicators of cancellation or unique identifiers.

**Datetime Features:** InvoiceDate is parsed and decomposed into Month, Day, and Hour.

**Categorical Encoding:** Country and StockCode are one-hot encoded with `pd.get_dummies`.

**Model:** XGBoost binary classifier with `max_depth=3`. Evaluated with 3-fold cross-validation using error rate as the metric.

## Project Structure

```
08_ecommerce_order_cancellation/
├── order_cancellation.py  # Full pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
xgboost
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `OnlineRetail-clean.csv` in the same directory and run:

```bash
python order_cancellation.py
```

The script prints the cross-validated accuracy and full CV results table.

## Notes

The dataset provides limited predictive signal for cancellation at order placement time. The model serves as a baseline; additional features such as customer order history would likely improve performance.
