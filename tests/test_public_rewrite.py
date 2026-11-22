import unittest

from rewrite import rewrite


class PublicRewriteTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "tables": {"customers": "clients", "orders": "purchases"},
            "columns": {
                "customers.customer_id": "clients.client_key",
                "customers.customer_name": "clients.full_name",
                "orders.order_id": "purchases.purchase_key",
                "orders.customer_id": "purchases.client_key",
                "orders.amount": "purchases.total",
            },
        }

    def test_unqualified_column(self):
        translated = rewrite(
            "SELECT customer_name FROM customers WHERE customer_id = 1",
            self.mapping,
        )
        self.assertIn('FROM "clients"', translated)
        self.assertIn('"full_name"', translated)
        self.assertIn('"client_key"', translated)

    def test_join(self):
        translated = rewrite(
            "SELECT c.customer_name, o.amount FROM customers AS c "
            "JOIN orders AS o ON c.customer_id = o.customer_id",
            self.mapping,
        )
        self.assertIn('"clients"', translated)
        self.assertIn('"purchases"', translated)
        self.assertIn('"full_name"', translated)
        self.assertIn('"total"', translated)

    def test_mapping_must_be_consistent(self):
        invalid = {
            "tables": {"customers": "clients"},
            "columns": {"customers.name": "wrong_table.full_name"},
        }
        with self.assertRaises(ValueError):
            rewrite("SELECT name FROM customers", invalid)

    def test_sqlite_identifier_case_is_ignored(self):
        mapping = {
            "tables": {"Singer": "Performer"},
            "columns": {"Singer.Name": "Performer.Full_Name"},
        }
        translated = rewrite("SELECT name FROM singer", mapping)
        self.assertIn('"Performer"', translated)
        self.assertIn('"Full_Name"', translated)

    def test_spider_double_quoted_string_literal(self):
        mapping = {
            "tables": {"AIRLINES": "airlines"},
            "columns": {
                "AIRLINES.Airline": "airlines.air_service",
                "AIRLINES.Country": "airlines.nation",
            },
        }
        translated = rewrite(
            'SELECT Country FROM AIRLINES WHERE Airline = "JetBlue Airways"',
            mapping,
        )
        self.assertIn("'JetBlue Airways'", translated)
        self.assertIn('"air_service"', translated)


if __name__ == "__main__":
    unittest.main()
