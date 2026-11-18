import unittest

from graphsignal.signals.resources import ResourceStore


class ResourceStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = ResourceStore()

    def test_update_and_export(self):
        self.store.update_resource(
            'process', tags={'process.pid': '1'},
            attributes={'process.command_line': 'python app.py'},
            first_seen_ts=10, last_seen_ts=20)

        exported = self.store.export()
        self.assertEqual(exported, [{
            'kind': 'process',
            'tags': {'process.pid': '1'},
            'attributes': {'process.command_line': 'python app.py'},
            'first_seen_ts': 10,
            'last_seen_ts': 20,
        }])

    def test_keyed_by_kind_and_tags(self):
        self.store.update_resource('process', tags={'process.pid': '1'},
                                   first_seen_ts=1, last_seen_ts=1)
        self.store.update_resource('process', tags={'process.pid': '2'},
                                   first_seen_ts=1, last_seen_ts=1)
        self.store.update_resource('host', tags={'process.pid': '1'},
                                   first_seen_ts=1, last_seen_ts=1)

        exported = self.store.export()
        self.assertEqual(len(exported), 3)

    def test_repeated_update_keeps_earliest_first_seen_and_latest_last_seen(self):
        self.store.update_resource('process', tags={'process.pid': '1'},
                                   first_seen_ts=10, last_seen_ts=20)
        self.store.update_resource('process', tags={'process.pid': '1'},
                                   first_seen_ts=15, last_seen_ts=30)
        self.store.update_resource('process', tags={'process.pid': '1'},
                                   first_seen_ts=5, last_seen_ts=25)

        exported = self.store.export()
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]['first_seen_ts'], 5)
        self.assertEqual(exported[0]['last_seen_ts'], 30)

    def test_none_kind_ignored(self):
        self.store.update_resource(None, tags={'t': '1'})
        self.assertEqual(self.store.export(), [])

    def test_default_timestamps_now(self):
        self.store.update_resource('host')
        exported = self.store.export()[0]
        self.assertGreater(exported['first_seen_ts'], 0)
        self.assertGreaterEqual(exported['last_seen_ts'],
                                exported['first_seen_ts'])

    def test_tag_and_attribute_truncation(self):
        self.store.update_resource(
            'process',
            tags={'k' * 100: 'v' * 500},
            attributes={'a' * 100: 'b' * 5000, 'none_attr': None})

        exported = self.store.export()[0]
        self.assertEqual(exported['tags'], {'k' * 50: 'v' * 250})
        self.assertEqual(exported['attributes'], {'a' * 50: 'b' * 2500})

    def test_engine_attributes_not_truncated(self):
        config = '{"settings": "' + 'x' * 20000 + '"}'
        self.store.update_resource(
            'engine',
            tags={'k' * 100: 'v' * 500},
            attributes={'engine.config': config})

        exported = self.store.export()[0]
        self.assertEqual(exported['attributes'], {'engine.config': config})
        # Tags are the resource's identity and stay capped for every kind.
        self.assertEqual(exported['tags'], {'k' * 50: 'v' * 250})

    def test_export_non_destructive(self):
        self.store.update_resource('process', tags={'process.pid': '1'},
                                   attributes={'a': '1'},
                                   first_seen_ts=1, last_seen_ts=2)

        first = self.store.export()
        second = self.store.export()
        self.assertEqual(first, second)

        first[0]['tags']['mutated'] = 'yes'
        first[0]['attributes']['mutated'] = 'yes'
        first[0]['kind'] = 'mutated'
        fresh = self.store.export()[0]
        self.assertEqual(fresh['kind'], 'process')
        self.assertEqual(fresh['tags'], {'process.pid': '1'})
        self.assertEqual(fresh['attributes'], {'a': '1'})


if __name__ == '__main__':
    unittest.main()
