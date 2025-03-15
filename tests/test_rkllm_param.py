import unittest
from rkllm_runtime.rkllm import ffi, RKLLMParam, RKLLMExtendParam


class TestRKLLMParam(unittest.TestCase):
    def test_from_defaults(self):
        model_path = b"/path/to/model.rkllm"
        rkllm_param = RKLLMParam.from_defaults()
        rkllm_param.model_path = ffi.new("char[]", model_path)

        self.assertEqual(ffi.string(rkllm_param.model_path), model_path)
        self.assertEqual(rkllm_param.max_context_len, 512)
        self.assertEqual(rkllm_param.max_new_tokens, -1)
        self.assertEqual(rkllm_param.skip_special_token, True)
        self.assertEqual(rkllm_param.top_k, 1)
        self.assertAlmostEqual(rkllm_param.top_p, 0.9)
        self.assertAlmostEqual(rkllm_param.temperature, 0.8)
        self.assertAlmostEqual(rkllm_param.repeat_penalty, 1.1)
        self.assertAlmostEqual(rkllm_param.frequency_penalty, 0.0)
        self.assertAlmostEqual(rkllm_param.presence_penalty, 0.0)
        self.assertEqual(rkllm_param.mirostat, 0)
        self.assertAlmostEqual(rkllm_param.mirostat_tau, 5.0)
        self.assertAlmostEqual(rkllm_param.mirostat_eta, 0.1)
        self.assertEqual(rkllm_param.is_async, False)
        self.assertEqual(ffi.string(rkllm_param.img_start), b"<img>")
        self.assertEqual(ffi.string(rkllm_param.img_end), b"</img>")
        self.assertEqual(ffi.string(rkllm_param.img_content), b"<unk>")
        self.assertEqual(rkllm_param.extend_param.base_domain_id, 0)

    def test_update_extend_param(self):
        model_path = b"/path/to/model.rkllm"
        base_domain_id = 12
        reserved = tuple(range(0, 224, 2))

        rkllm_param = RKLLMParam.from_defaults()
        rkllm_extend_param = RKLLMExtendParam(
            base_domain_id=base_domain_id, reserved=reserved
        )
        rkllm_param.extend_param = rkllm_extend_param[0]  # type: ignore[index]

        self.assertEqual(rkllm_param.extend_param.base_domain_id, base_domain_id)
        self.assertEqual(tuple(rkllm_param.extend_param.reserved), reserved)


if __name__ == "__main__":
    unittest.main()
