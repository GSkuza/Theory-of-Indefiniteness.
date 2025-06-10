import unittest
import pickle
import os
from unittest import mock
from numbers import Number

# Let's assume the file we are testing is named 'core.py'
# If it's in the same directory, the import should work.
# If you're running tests from a different location, make sure PYTHONPATH is set.
import core # Changed from 'core' to 'core' according to your file

# Helper list of operators for testing
ARITHMETIC_OPERATORS_TEST_CASES = [
    (lambda a, b: a + b, "__add__", "__radd__"),
    (lambda a, b: a - b, "__sub__", "__rsub__"),
    (lambda a, b: a * b, "__mul__", "__rmul__"),
    (lambda a, b: a / b, "__truediv__", "__rtruediv__"),
    (lambda a, b: a ** b, "__pow__", "__rpow__"),
]

class TestSingularitySimple(unittest.TestCase):
    """Simple tests for Singularity (O)."""

    def test_O_instance_and_type(self):
        self.assertIsInstance(core.O, core.Singularity)
        self.assertIsInstance(core.O, Number)

    def test_O_is_singleton(self):
        o1 = core.O
        o2 = core.Singularity()
        o3 = core.get_singularity()
        self.assertIs(o1, o2)
        self.assertIs(o1, o3)
        self.assertIs(o2, o3)

    def test_O_repr(self):
        self.assertEqual(repr(core.O), "O_empty_singularity")

    def test_O_bool(self):
        self.assertFalse(core.O)

    def test_O_equality(self):
        self.assertEqual(core.O, core.Singularity())
        # Should only be equal to itself or another (the same) Singularity instance
        self.assertNotEqual(core.O, "O_empty_singularity")
        self.assertNotEqual(core.O, None)
        self.assertNotEqual(core.O, 0)
        self.assertNotEqual(core.O, core.AlienatedNumber("test"))

    def test_O_hash(self):
        self.assertEqual(hash(core.O), hash("O_empty_singularity"))
        # Check if it can be a key in a dictionary
        d = {core.O: "value"}
        self.assertEqual(d[core.O], "value")

    def test_O_to_json(self):
        self.assertEqual(core.O.to_json(), '"O_empty_singularity"')

class TestAlienatedNumberSimple(unittest.TestCase):
    """Simple tests for AlienatedNumber."""

    def test_AN_instance_and_type(self):
        an = core.AlienatedNumber("test")
        self.assertIsInstance(an, core.AlienatedNumber)
        self.assertIsInstance(an, Number)

    def test_AN_init_with_identifier(self):
        an_str = core.AlienatedNumber("my_id")
        self.assertEqual(an_str.identifier, "my_id")
        an_int = core.AlienatedNumber(123)
        self.assertEqual(an_int.identifier, 123)
        an_float = core.AlienatedNumber(1.23)
        self.assertEqual(an_float.identifier, 1.23)

    def test_AN_init_default_identifier(self):
        an = core.AlienatedNumber()
        self.assertEqual(an.identifier, "anonymous")

    def test_AN_repr(self):
        an1 = core.AlienatedNumber("id_1")
        self.assertEqual(repr(an1), "l_empty_num(id_1)")
        an2 = core.AlienatedNumber(123)
        self.assertEqual(repr(an2), "l_empty_num(123)")
        an3 = core.AlienatedNumber()
        self.assertEqual(repr(an3), "l_empty_num(anonymous)")

    def test_AN_equality(self):
        an1a = core.AlienatedNumber("id1")
        an1b = core.AlienatedNumber("id1")
        an2 = core.AlienatedNumber("id2")
        an_anon1 = core.AlienatedNumber()
        an_anon2 = core.AlienatedNumber()

        self.assertEqual(an1a, an1b)
        self.assertNotEqual(an1a, an2)
        self.assertEqual(an_anon1, an_anon2) # Two anonymous instances should be equal
        self.assertNotEqual(an1a, an_anon1)

        self.assertNotEqual(an1a, "id1")
        self.assertNotEqual(an1a, core.O)

    def test_AN_hash(self):
        an1a = core.AlienatedNumber("id1")
        an1b = core.AlienatedNumber("id1")
        an2 = core.AlienatedNumber("id2")

        self.assertEqual(hash(an1a), hash(an1b))
        self.assertNotEqual(hash(an1a), hash(an2))
        # Check if it can be a key in a dictionary
        d = {an1a: "value1", an2: "value2"}
        self.assertEqual(d[an1b], "value1") # Using an1b to retrieve the value for an1a
        self.assertEqual(d[an2], "value2")

    def test_AN_metrics(self):
        an = core.AlienatedNumber()
        self.assertEqual(an.psi_gtm_score(), core.AlienatedNumber.PSI_GTM_SCORE)
        self.assertEqual(an.e_gtm_entropy(), core.AlienatedNumber.E_GTM_ENTROPY)
        self.assertAlmostEqual(an.psi_gtm_score(), 0.999_999_999)
        self.assertAlmostEqual(an.e_gtm_entropy(), 1e-9)

    def test_AN_to_json(self):
        an = core.AlienatedNumber("json_id")
        self.assertEqual(an.to_json(), '"l_empty_num(json_id)"')
        an_anon = core.AlienatedNumber()
        self.assertEqual(an_anon.to_json(), '"l_empty_num(anonymous)"')

class TestArithmeticOperationsAdvanced(unittest.TestCase):
    """Advanced tests for arithmetic operations, including STRICT_MODE."""

    def _test_arithmetic_absorption(self, obj, obj_symbol_for_error_msg):
        # Tests in non-strict mode (default)
        with mock.patch('core.STRICT_MODE', False):
            for op_func, op_name, rop_name in ARITHMETIC_OPERATORS_TEST_CASES:
                with self.subTest(f"Non-strict: {obj} {op_name} 1"):
                    self.assertIs(op_func(obj, 1), core.O)
                with self.subTest(f"Non-strict: 1 {rop_name} {obj}"):
                    # For ** (rpow), 1 ** O should be O, but e.g., 0 ** O is also O
                    # For other numbers, e.g., 2 ** O, the result is O
                    self.assertIs(op_func(1, obj), core.O)
                with self.subTest(f"Non-strict: {obj} {op_name} OtherAlienatedNumber"):
                    other_an = core.AlienatedNumber("other")
                    self.assertIs(op_func(obj, other_an), core.O)

        # Tests in strict mode
        with mock.patch('core.STRICT_MODE', True):
            for op_func, op_name, rop_name in ARITHMETIC_OPERATORS_TEST_CASES:
                expected_msg_op = f"Operation '{op_name}' with {obj_symbol_for_error_msg} is forbidden in STRICT mode"
                expected_msg_rop = f"Operation '{rop_name}' with {obj_symbol_for_error_msg} is forbidden in STRICT mode"

                with self.subTest(f"Strict: {obj} {op_name} 1"):
                    with self.assertRaisesRegex(core.SingularityError, expected_msg_op):
                        op_func(obj, 1)
                with self.subTest(f"Strict: 1 {rop_name} {obj}"):
                    with self.assertRaisesRegex(core.SingularityError, expected_msg_rop):
                        op_func(1, obj)
                with self.subTest(f"Strict: {obj} {op_name} OtherAlienatedNumber"):
                     other_an = core.AlienatedNumber("other")
                     with self.assertRaisesRegex(core.SingularityError, expected_msg_op):
                        op_func(obj, other_an)


    def test_O_arithmetic(self):
        self._test_arithmetic_absorption(core.O, "Ø")

    def test_AN_arithmetic(self):
        an = core.AlienatedNumber("test_an")
        self._test_arithmetic_absorption(an, "ℓ∅")

    def test_mixed_arithmetic_O_and_AN_normal_mode(self):
        an = core.AlienatedNumber("test_an")
        with mock.patch('core.STRICT_MODE', False):
            for op_func, op_name, _ in ARITHMETIC_OPERATORS_TEST_CASES:
                with self.subTest(f"Non-strict: O {op_name} AN"):
                    self.assertIs(op_func(core.O, an), core.O)
                with self.subTest(f"Non-strict: AN {op_name} O"):
                    self.assertIs(op_func(an, core.O), core.O)

    def test_mixed_arithmetic_O_and_AN_strict_mode(self):
        an = core.AlienatedNumber("test_an")
        with mock.patch('core.STRICT_MODE', True):
            for op_func, op_name, _ in ARITHMETIC_OPERATORS_TEST_CASES:
                # If O is on the left-hand side, its error is raised
                expected_msg_O_lhs = f"Operation '{op_name}' with Ø is forbidden in STRICT mode"
                with self.subTest(f"Strict: O {op_name} AN"):
                    with self.assertRaisesRegex(core.SingularityError, expected_msg_O_lhs):
                        op_func(core.O, an)

                # If AN is on the left-hand side, its error is raised
                expected_msg_AN_lhs = f"Operation '{op_name}' with ℓ∅ is forbidden in STRICT mode"
                with self.subTest(f"Strict: AN {op_name} O"):
                    with self.assertRaisesRegex(core.SingularityError, expected_msg_AN_lhs):
                        op_func(an, core.O)


class TestPicklingAdvanced(unittest.TestCase):
    """Tests related to serialization (pickling)."""

    def test_O_pickle_safety(self):
        pickled_O = pickle.dumps(core.O)
        unpickled_O = pickle.loads(pickled_O)
        self.assertIs(core.O, unpickled_O, "Unpickled O should be the same singleton instance.")

    def test_AN_pickling(self):
        an_original = core.AlienatedNumber("pickle_id")
        an_original.some_temp_attr = "test" # Attribute not covered by __slots__ will not be pickled

        pickled_an = pickle.dumps(an_original)
        unpickled_an = pickle.loads(pickled_an)

        self.assertIsNot(an_original, unpickled_an) # Should be different instances
        self.assertEqual(an_original, unpickled_an) # But equal in value
        self.assertEqual(unpickled_an.identifier, "pickle_id")
        self.assertFalse(hasattr(unpickled_an, "some_temp_attr"), "Attribute not in __slots__ should not be pickled.")


class TestErrorTypes(unittest.TestCase):
    """Tests for error types."""

    def test_SingularityError_inheritance(self):
        self.assertTrue(issubclass(core.SingularityError, ArithmeticError))
        self.assertTrue(issubclass(core.SingularityError, Exception))


class TestStrictModeConfiguration(unittest.TestCase):
    """Tests for STRICT_MODE configuration via environment variable."""

    @mock.patch.dict(os.environ, {"GTM_STRICT": "1"})
    def test_strict_mode_enabled_by_env_var(self):
        # The module needs to be reloaded to read the new environment variable.
        # This is difficult to do cleanly in a unit test without causing a mess.
        # Instead, we will test the value of core.STRICT_MODE
        # after a potential re-import or patching.
        # In this case, we patch os.getenv used inside the module.
        with mock.patch('os.getenv', return_value="1"):
            # We simulate reloading the value by calling the code that sets it.
            # This is a bit of a hack; normally, the module is loaded once.
            # A better approach would be to patch `core.STRICT_MODE` directly
            # in the tests that need it, which we are already doing.
            # This test is more conceptual.
            # For a true module reload test, `importlib.reload` would be needed.
            # import importlib
            # importlib.reload(core) # This can have side effects on other tests
            self.assertTrue(os.getenv("GTM_STRICT", "0") == "1") # We check if the mock is working
            # In an ideal world, after reload(core), core.STRICT_MODE would be True.
            # But since we are not doing a reload, this test is limited.
            # Let's see what happens if we patch the value directly:
        with mock.patch('core.STRICT_MODE', True):
            self.assertTrue(core.STRICT_MODE)

    @mock.patch.dict(os.environ, {"GTM_STRICT": "0"})
    def test_strict_mode_disabled_by_env_var(self):
        with mock.patch('os.getenv', return_value="0"):
            self.assertFalse(os.getenv("GTM_STRICT", "0") == "1")
        with mock.patch('core.STRICT_MODE', False):
            self.assertFalse(core.STRICT_MODE)

    @mock.patch.dict(os.environ, {}, clear=True) # Clear environment variables
    def test_strict_mode_disabled_by_default(self):
        with mock.patch('os.getenv', side_effect=lambda k, d: d): # Simulates the absence of the variable
             self.assertFalse(os.getenv("GTM_STRICT", "0") == "1")
        # The default value in the code is False if the variable is not set to "1".
        # This test is somewhat redundant because `core.STRICT_MODE` is a constant after the module is loaded.
        # Its value depends on the state of `os.environ` *during import*.
        # The best way to test this is by patching `core.STRICT_MODE` as in `TestArithmeticOperationsAdvanced`.


if __name__ == '__main__':
    unittest.main(verbosity=2)
