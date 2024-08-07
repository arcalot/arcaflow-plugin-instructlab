#!/usr/bin/env python3
import unittest

import instructlab_plugin
from arcaflow_plugin_sdk import plugin

import instructlab_plugin_schema


class HelloWorldTest(unittest.TestCase):
    @staticmethod
    def test_serialization():
        plugin.test_object_serialization(instructlab_plugin.InputParams(
            pytorch_args=instructlab_plugin_schema.TorchrunArgs(),
            training_args=instructlab_plugin_schema.TrainingArgs(
                model_path="",
                data_path="",
            ),
        ))

        plugin.test_object_serialization(
            instructlab_plugin.SuccessOutput("Hello, world!")
        )

        plugin.test_object_serialization(
            instructlab_plugin.ErrorOutput(error="This is an error")
        )


if __name__ == "__main__":
    unittest.main()
