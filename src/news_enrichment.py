import gc
import logging
import time

import docker
import dspy
import fasttext
import pycountry
import torch
from huggingface_hub import hf_hub_download
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class Processor:
    def process_batch(self, data: list[dict]):
        for el in data:
            self.process(el)

    def process(self, data: dict):
        pass

    def setup(self):
        pass

    def shutdown(self):
        torch.cuda.empty_cache()
        gc.collect()


def three_to_two_letter(lang_code: str) -> str | None:
    if lang_code == "arb":
        return "ar"
    if lang_code == "zsm":
        return "ms"
    if lang_code == "npi":
        return "ne"
    try:
        language = pycountry.languages.get(alpha_3=lang_code)
        rval = language.alpha_2  # Convert to 2-letter code
    except AttributeError:
        rval = None  # If no 2-letter code exists
    except KeyError:
        rval = None
    if rval is None:
        logging.warning("Cannot convert language code: {lang_code=} {language=}")
    return rval


class LanguageIdentifier(Processor):
    def __init__(self):
        self.model = None

    def setup(self):
        model_name = "facebook/fasttext-language-identification"
        file_name = "model.bin"
        self.model = fasttext.load_model(hf_hub_download(model_name, file_name))

    def process(self, data):
        prediction = self.predict(data["title"])
        if prediction == "ko":
            prediction = data["lang"]
        data["lang/fasttext"] = prediction

    def predict(self, text) -> str | None:
        if self.model is None:
            self.setup()
        prediction = self.model.predict([text])[0][0][0]
        logging.info("LanguageIdentifier prediction %s", prediction)
        three_letter = prediction.rsplit("_")[-2]
        return three_to_two_letter(three_letter)

    def shutdown(self):
        del self.model
        self.model = None
        super().shutdown()


class TitleArbitrator(Processor):
    def __init__(self):
        pass

    def setup(self):
        pass

    def process(self, data):
        data["title/arbitrated"] = (
            data["title"]
            if data["lang/fasttext"] == "en"
            else data.get("title/translated")
        )

    def shutdown(self):
        pass


class SeedTranslator(Processor):
    language_names = {
        "ar": "Arabic",
        "cs": "Czech",
        "da": "Danish",
        "de": "German",
        "es": "Spanish",
        "fi": "Finnish",
        "fr": "French",
        "hr": "Croatian",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "ms": "Malay",
        "nb": "Norwegian Bokmal",
        "nl": "Dutch",
        "no": "Norwegian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sv": "Swedish",
        "th": "Thai",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "vi": "Vietnamese",
        "zh": "Chinese",
        "zh-Hant": "Chinese",
    }

    def __init__(
        self,
        hf_token: str,
        hf_home: str,
        vllm_openai_api_key: str,
        vllm_openai_api_base: str,
    ):
        self.docker_client = docker.from_env()
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=vllm_openai_api_key,
            base_url=vllm_openai_api_base,
        )
        self.container = None
        self.container_name = "seed_translate"
        self.hf_token = hf_token
        self.hf_home = hf_home
        self.model_name = "ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8"
        # self.model_name = "ByteDance-Seed/Seed-X-PPO-7B-AWQ-Int4"

    def setup_vllm(self):
        self.container_name = "seed_translate_vllm"
        with_name = self.docker_client.containers.list(
            all=True, filters={"name": self.container_name}
        )
        if len(with_name) > 0:
            self.container = with_name[0]
            if self.container.status != "running":
                self.container.restart()
        else:
            self.container = self.docker_client.containers.run(
                "vllm/vllm-openai:latest",
                name=self.container_name,
                command=f"--model {self.model_name} --gpu-memory-utilization=0.7",
                ipc_mode="host",
                runtime="nvidia",
                environment={"HUGGING_FACE_HUB_TOKEN": self.hf_token},
                ports={"8000": 8000},
                volumes=[
                    f"{self.hf_home}:/root/.cache/huggingface",
                ],
                detach=True,
                device_requests=[
                    docker.types.DeviceRequest(
                        count=-1,
                        capabilities=[["gpu"]],
                    )
                ],
            )  # type: ignore
        # TODO wait until ready
        while True:
            try:
                self.client.models.list()
                break
            except:  # noqa
                time.sleep(1)

    def setup_sglang(self):
        self.container_name = "seed_translate_sglang"
        with_name = self.docker_client.containers.list(
            all=True, filters={"name": self.container_name}
        )
        if len(with_name) > 0:
            self.container = with_name[0]
            if self.container.status != "running":
                self.container.restart()
        else:
            self.container = self.docker_client.containers.run(
                "lmsysorg/sglang:latest",
                name=self.container_name,
                command=f"python3 -m sglang.launch_server --port 8000 --model-path {self.model_name}",
                ipc_mode="host",
                runtime="nvidia",
                environment={"HUGGING_FACE_HUB_TOKEN": self.hf_token},
                ports={"8000": 8000},
                volumes=[
                    f"{self.hf_home}:/root/.cache/huggingface",
                ],
                detach=True,
                device_requests=[
                    docker.types.DeviceRequest(
                        count=-1,
                        capabilities=[["gpu"]],
                    )
                ],
            )  # type: ignore
        # TODO wait until ready
        while True:
            try:
                self.client.models.list()
                break
            except:
                time.sleep(1)

    def setup(self):
        self.setup_vllm()

    def shutdown(self):
        if self.container is not None:
            with_name = self.docker_client.containers.list(
                all=True, filters={"name": self.container_name}
            )
            if len(with_name) > 0:
                self.container = with_name[0]
                if self.container.status == "running":
                    self.container.stop()

    @staticmethod
    def supported_languages():
        return list(SeedTranslator.language_names.keys())

    def process(self, data):
        if data.get("lang/fasttext") in self.supported_languages():
            data["title/translated"] = self.translate(
                data["title"], data["lang/fasttext"]
            )

    def process_batch(self, data):
        indexes = []
        batch = []
        for idx, d in enumerate(data):
            if d.get("lang/fasttext") in self.supported_languages():
                indexes.append(idx)
                batch.append((d["title"], d["lang/fasttext"]))
        if len(batch) == 0:
            return
        translations = self.translate_batch(batch)
        for tidx, translation in zip(indexes, translations):
            data[tidx]["title/translated"] = translation

    def translate(self, input: str, input_language: str) -> str:
        language_name = self.language_names[input_language]

        models = self.client.models.list()
        model = models.data[0].id

        # Completion API
        completion = self.client.completions.create(
            model=model,
            prompt=f"Translate the following {language_name} sentence into English:\n{input} <en>",
            echo=False,
            n=1,
            stream=False,
            max_tokens=512,
        )

        return completion.choices[0].text.strip()

    def prompt_and_translate(self, input: str, input_language: str) -> dict:
        language_name = self.language_names[input_language]

        models = self.client.models.list()
        model = models.data[0].id

        prompt = f"Translate the following {language_name} sentence into English:\n{input} <en>"

        # Completion API
        completion = self.client.completions.create(
            model=model,
            prompt=prompt,
            echo=False,
            n=1,
            stream=False,
            max_tokens=512,
        )

        return {"prompt": prompt, "completion": completion.choices[0].text.strip()}

    def translate_batch(self, inputs: list[tuple[str, str]]) -> list[str]:
        models = self.client.models.list()
        model = models.data[0].id

        prompts = []
        for text, lang in inputs:
            language_name = self.language_names[lang]
            prompts.append(
                f"Translate the following {language_name} sentence into English:\n{text} <en>"
            )

        # Completion API
        completion = self.client.completions.create(
            model=model,
            prompt=prompts,
            echo=False,
            n=1,
            stream=False,
            max_tokens=1024,
        )

        return [choice.text.strip() for choice in completion.choices]


class BoilerplateRemover(Processor):
    class BoilerplateRemovalInfo(dspy.Signature):
        """Strip metadata, in particular the source name (if present), from news headlines"""

        headline = dspy.InputField(desc="A possible noisy headline")
        source_domain = dspy.InputField(desc="The domain of the news source")
        source_name = dspy.InputField(desc="The name of the news source")
        clean_headline = dspy.OutputField(desc="Clean headline")

    def __init__(
        self,
        hf_token: str,
        hf_home: str,
        vllm_openai_api_base: str,
        vllm_openai_api_key: str,
    ):
        self.docker_client = docker.from_env()
        self.container_name = "vllm_boilerplate"
        self.container = None
        self.hf_token = hf_token
        self.hf_home = hf_home
        self.vllm_openai_api_key = vllm_openai_api_key
        self.vllm_openai_api_base = vllm_openai_api_base
        self.module = None
        self.lm = None
        # self.model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
        # self.model_name = "unsloth/Qwen3-0.6B-bnb-4bit"
        self.model_name = "Qwen/Qwen3-0.6B"

        # just use this to easily wait for container ready
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=vllm_openai_api_key,
            base_url=vllm_openai_api_base,
        )

    def setup(self):
        with_name = self.docker_client.containers.list(
            filters={"name": self.container_name}
        )
        if len(with_name) > 0:
            self.container = with_name[0]
            if self.container.status != "running":
                self.container.restart()
        else:
            self.container = self.docker_client.containers.run(
                "vllm/vllm-openai:latest",
                name=self.container_name,
                command=f"--model {self.model_name} --gpu-memory-utilization=0.7",
                ipc_mode="host",
                runtime="nvidia",
                environment={"HUGGING_FACE_HUB_TOKEN": self.hf_token},
                ports={"8000": 8000},
                volumes=[
                    f"{self.hf_home}:/root/.cache/huggingface",
                    "/home/nrg/.cache/vllm/torch_compile_cache/:/root/.cache/vllm/torch_compile_cache/",
                ],
                detach=True,
                device_requests=[
                    docker.types.DeviceRequest(
                        count=-1,
                        capabilities=[["gpu"]],
                    )
                ],
            )  # type: ignore
        # TODO wait until ready
        while True:
            try:
                self.client.models.list()
                break
            except:
                time.sleep(1)

        self.lm = dspy.LM(
            f"hosted_vllm/{self.model_name}",
            api_base=self.vllm_openai_api_base,
            api_key=self.vllm_openai_api_key,
        )
        dspy.settings.configure(lm=self.lm)
        self.module = dspy.Predict(BoilerplateRemover.BoilerplateRemovalInfo)

    def shutdown(self):
        if self.container is not None:
            self.container.stop()
        if self.lm is not None:
            del self.lm
            self.lm = None
        if self.module is not None:
            del self.module
            self.module = None

    def process(self, data):
        # if data.get("lang/fasttext") in self.supported_languages():
        #    data["title/translated"] = self.translate(
        #        data["title"], data["lang/fasttext"]
        #    )
        if (title := data.get("title/arbitrated")) is not None:
            data["title/clean"] = self.strip_boilerplate(
                headline=title,
                source_domain=data["domain"],
                source_name=data["outletName"],
            )

    def strip_boilerplate(
        self, headline: str, source_domain: str, source_name: str
    ) -> str:
        if self.module is None:
            self.setup()
        assert self.module is not None
        return self.module(
            headline=headline, source_domain=source_domain, source_name=source_name
        ).clean_headline

    def requires(self) -> set[str]:
        return {"title/arbitrated"}

    def provides(self) -> set[str]:
        return {"title/clean"}


class ENPipelineClassifier(Processor):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.pipe = None
        self.pipe_kwargs = kwargs

    def setup(self):
        self.pipe = pipeline(
            "text-classification", model=self.model_name, **self.pipe_kwargs
        )

    def process(self, data):
        if self.pipe is None:
            self.setup()
        assert self.pipe is not None
        data[f"clf/{self.model_name}"] = self.pipe([data["title/arbitrated"]])

    def shutdown(self):
        del self.pipe
        self.pipe = None
        super().shutdown()


class ZeroShotClassifier(Processor):
    def __init__(
        self,
        model_name="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        class_name="international politics",
        **kwargs,
    ):
        self.model_name = model_name
        self.class_name = class_name
        self.class_name_key = class_name.lower().replace(" ", "_")
        self.pipe = None
        self.pipe_kwargs = kwargs

    def setup(self):
        self.pipe = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        )

    def process(self, data):
        if (title := data.get("title/arbitrated")) is not None:
            data[f"zs_clf/{self.class_name_key}"] = self.predict(title)

    def process_batch(self, data):
        indices = []
        titles = []
        for idx, el in enumerate(data):
            if (title := el.get("title/arbitrated")) is not None:
                titles.append(title)
                indices.append(idx)
        scores = self.predict_many(titles)
        for idx, score in zip(indices, scores):
            data[idx][f"zs_clf/{self.class_name_key}"] = score

    def predict(self, text):
        if len(text.strip()) == 0:
            return 0
        if self.pipe is None:
            self.setup()
        assert self.pipe is not None
        hypothesis_template = "This text is about {}"
        classes_verbalized = [self.class_name]
        output = self.pipe(
            text,
            classes_verbalized,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        return output["scores"][0]

    def predict_many(self, texts):
        if self.pipe is None:
            self.setup()
        assert self.pipe is not None
        hypothesis_template = "This text is about {}"
        classes_verbalized = [self.class_name]
        output = self.pipe(
            texts,
            classes_verbalized,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        return [o["scores"][0] for o in output]

    def shutdown(self):
        del self.pipe
        self.pipe = None
        super().shutdown()

    def requires(self) -> set[str]:
        return {"title/arbitrated"}

    def provides(self) -> set[str]:
        return {f"zs_clf/{self.class_name_key}"}


class MultilingualZeroShotClassifier(Processor):
    def __init__(
        self,
        model_name="MoritzLaurer/bge-m3-zeroshot-v2.0",
        class_name="international politics",
        **kwargs,
    ):
        self.model_name = model_name
        self.class_name = class_name
        self.class_name_key = class_name.lower().replace(" ", "_")
        self.pipe = None
        self.pipe_kwargs = kwargs

    def setup(self):
        self.pipe = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/bge-m3-zeroshot-v2.0",
        )

    def process(self, data):
        if (title := data.get("title")) is not None:
            data[f"ml_zs_clf/{self.class_name_key}"] = self.predict(title)

    def predict(self, text):
        if len(text.strip()) == 0:
            return 0
        if self.pipe is None:
            self.setup()
        assert self.pipe is not None
        hypothesis_template = "This text is about {}"
        classes_verbalized = [self.class_name]
        output = self.pipe(
            text,
            classes_verbalized,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        return output["scores"][0]

    def shutdown(self):
        del self.pipe
        self.pipe = None
        super().shutdown()

    def requires(self) -> set[str]:
        return {"title"}

    def provides(self) -> set[str]:
        return {f"ml_zs_clf/{self.class_name_key}"}


class Embedding(Processor):
    def __init__(self, embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model = None
        self.embedding_model_name = embedding_model_name

    def setup(self):
        self.model = SentenceTransformer(
            self.embedding_model_name,
            model_kwargs=dict(dtype=torch.bfloat16),
            device="cuda",
        )

    def embed(self, sentences: list[str]) -> list[list[float]]:
        if self.model is None:
            self.setup()
        assert self.model is not None
        return self.model.encode(sentences, show_progress_bar=False).tolist()

    def process(self, data):
        if (title := data.get("title/arbitrated")) is not None:
            data["embedding"] = self.embed([title])[0]

    def shutdown(self):
        del self.model
        self.model = None
        super().shutdown()
