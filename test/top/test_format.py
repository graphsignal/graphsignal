import unittest

from graphsignal.top import format as fmt


class FormatTest(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(fmt.fmt_bytes(None), '-')
        self.assertEqual(fmt.fmt_bytes(512), '512 B')
        self.assertEqual(fmt.fmt_bytes(98_100_000_000), '98.1 GB')
        self.assertEqual(fmt.fmt_bytes(1_250_000_000_000), '1.2 TB')

    def test_rate(self):
        self.assertEqual(fmt.fmt_rate(None), '-')
        self.assertEqual(fmt.fmt_rate(0), '0')
        self.assertEqual(fmt.fmt_rate(980), '980')
        self.assertEqual(fmt.fmt_rate(1840), '1.84k')
        self.assertEqual(fmt.fmt_rate(11_200), '11.2k')
        self.assertEqual(fmt.fmt_rate(2_500_000), '2.50M')

    def test_count(self):
        self.assertEqual(fmt.fmt_count(1980), '1,980')
        self.assertEqual(fmt.fmt_count(None), '-')

    def test_duration(self):
        self.assertEqual(fmt.fmt_duration_ns(None), '-')
        self.assertEqual(fmt.fmt_duration_ns(400), '400 ns')
        self.assertEqual(fmt.fmt_duration_ns(158_600), '159 µs')
        self.assertEqual(fmt.fmt_duration_ns(58_600), '58.6 µs')
        self.assertEqual(fmt.fmt_duration_ns(24_100_000), '24.1 ms')
        self.assertEqual(fmt.fmt_duration_ns(2_900_000_000), '2.9 s')
        self.assertEqual(fmt.fmt_seconds(0.182), '182 ms')

    def test_uptime_and_window(self):
        self.assertEqual(fmt.fmt_uptime(8077), '02:14:37')
        self.assertEqual(fmt.fmt_uptime(90_000), '1d 01:00')
        self.assertEqual(fmt.fmt_window(45), '45s')
        self.assertEqual(fmt.fmt_window(120), '2m')
        self.assertEqual(fmt.fmt_window(7200), '2h')

    def test_sparkline_scales_and_right_aligns(self):
        line = fmt.sparkline([0, 50, 100], 5, vmax=100)
        self.assertEqual(len(line), 5)
        self.assertEqual(line[:2], '  ')
        self.assertEqual(line[2], fmt.SPARK_BLOCKS[1])
        self.assertEqual(line[4], fmt.SPARK_BLOCKS[-1])

    def test_sparkline_none_is_blank_and_zero_is_floor(self):
        self.assertEqual(fmt.sparkline([None, 0, 0], 3), ' ' + fmt.SPARK_BLOCKS[1] * 2)
        self.assertEqual(fmt.sparkline([], 3), '   ')
        self.assertEqual(fmt.sparkline([1, 2, 3], 0), '')

    def test_bar(self):
        self.assertEqual(fmt.bar(0.5, 4), '▇▇▁▁')
        self.assertEqual(fmt.bar(None, 3), '▁▁▁')
        self.assertEqual(fmt.bar(2.0, 2), '▇▇')

    def test_sparkline_never_fills_the_cell(self):
        self.assertNotIn('█', fmt.sparkline([100, 50], 2, vmax=100))
        self.assertEqual(fmt.sparkline([100], 1, vmax=100), '▇')

    def test_fit_and_pad(self):
        self.assertEqual(fmt.fit('hello', 10), 'hello')
        self.assertEqual(fmt.fit('hello world', 5), 'hell…')
        self.assertEqual(fmt.fit('x', 0), '')
        self.assertEqual(fmt.pad('ab', 4), 'ab  ')
        self.assertEqual(fmt.pad('ab', 4, align='right'), '  ab')
        self.assertEqual(fmt.pad('abcdef', 4), 'abc…')
