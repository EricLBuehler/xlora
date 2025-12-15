# X-LoRA
LoRA Uzmanları Karışımı (Mixture of LoRA Experts): Uzmanların karışımı veya MoE tekniğini kullanarak, ince ayar yapılmış (fine-tuned) LoRA uzmanlarının gücünden yararlanın.

X-LoRA, LoRA adaptörleri için ölçekleme değerlerini öğrenerek çalışır. Öğrenilen bu ölçekleme değerleri, LoRA uzmanlarını yoğun (dense) bir biçimde yönetmek (gating) için kullanılır. Ayrıca, tüm LoRA adaptörleri ve temel model dondurulmuştur; bu durum, düşük parametre sayısı sayesinde verimli bir ince ayar (fine-tuning) yapılmasına olanak tanır.

X-LoRA, herhangi bir HuggingFace Transformers modeline kolayca uygulanabilir. Lütfen ağırlıklarımızı inceleyin, [burdan](https://huggingface.co/lamm-mit/x-lora) ve bizim [dökümantasyon](https://arxiv.org/abs/2402.07148).

### Token bazlı ölçeklendirmeler
![Token bazlı ölçeklendirmeler](./res/token_by_token_scalings.gif)

## Avantajlar ve Özellikler

- Etkili: Uzmanların yoğun (dense) bir şekilde yönlendirilmesi (gating), etkili bir harmanlama/karışım sağlar.
- Verimli İnce Ayar (Fine-tuning): Düşük eğitilebilir parametre sayısı.
- Hiyerarşik Kapsülleme Stratejisi: Biyolojiden esinlenen bir strateji izleyerek; mevcut eğitilmiş modelleri veya model bölümlerini yeniden kullanır ve bunları, birden fazla uzmanı ilgilendiren karmaşık görevleri çözmek için kullanır.
- Kullanımı Kolay API: `add_xlora_to_model` fonksiyonu ve geniş uyumluluk.
- LoRA Adaptörlerini Dinamik Olarak Karıştırma: Adaptörlerin derinlemesine ve katman bazlı (layer-wise) kombinasyonları.

### Mimari
<p align="center">
    <img src="./res/general_arch_v5.png" alt="General Architecture" width=75%/>
</p>

X-LoRA'ya nasıl başlayacağınıza dair bazı örnekler için [örnekler](examples) klasörüne bakın.

## Verimli Çıkarım Desteği
[Mistral.rs](https://github.com/EricLBuehler/mistral.rs) X-LoRA'yı destekleyen bir çıkarım çerçevesidir! Kullanmak için kurulum talimatlarını izleyin ve bir X-LoRA çıkarım platformunu başlatmak üzere aşağıdaki komutu çalıştırın.

`./mistralrs-server --port 1234 x-lora-mistral -o ordering.json`

Kendi modellerinizi kullanmak için Temel ve X-LoRA Huggingface model kimlikleri (ID'leri), komut satırı anahtarları aracılığıyla belirtilebilir. Daha fazla ayrıntı için lütfen Github sayfasına bakınız.

## Kurulum
Pip sürümü yayınlanana kadar, X-LoRA'yı yüklemek için aşağıdaki komutu çalıştırın.
`pip install git+https://github.com/EricLBuehler/xlora.git`

## Examples
Örnekler [bu](./examples/simple.ipynb) örnekten alıntı.

- [Converting a model](README.md#converting-a-model)
- [Loading a trained X-LoRA model from scratch](README.md#loading-a-trained-x-lora-model-from-scratch)
- [Loading a trained X-LoRA model with a convenience function](README.md#loading-a-trained-x-lora-model-with-a-convenience-function)
- [Scalings logging](README.md#scalings-logging)
- [Trainable parameters](README.md#trainable-parameters)
- [Setting trainability of adapters dynamically](README.md#setting-trainability-of-adapters-dynamically)
- [Setting and resetting the scaling pass value](README.md#setting-and-resetting-the-scaling-pass-value)
- [Setting and getting the global LoRA weight](README.md#setting-and-getting-the-global-lora-weight)
- [Setting and getting the top-k lora value](README.md#setting-and-getting-the-top-k-lora-value)

### Model Dönüştürmek
```python
import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM # type: ignore

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

config = AutoConfig.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="auto",
)

### Convert the model to X-LoRA
model_created = xlora.add_xlora_to_model(
    model=model,
    xlora_config=xlora.xLoRAConfig(
        config.hidden_size,
        base_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        xlora_depth=8,
        device=torch.device("cuda"),
        adapters={
            "adapter_1": "./path/to/the/checkpoint/",
            "adapter_2": "./path/to/the/checkpoint/",
            "adapter_n": "./path/to/the/checkpoint/",
        },
    ),
    verbose=True,
)
```
### Eğitilmiş bir X-LoRA modelini sıfırdan yükleme
```python
import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM # type: ignore

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

config = AutoConfig.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="auto",
)

model_created = xlora.from_pretrained(
    "./path/to/saved/model",
    model,
    "cuda",
)
```

### Eğitilmiş bir X-LoRA modelini yardımcı (kolaylaştırıcı) bir fonksiyonla yükleme
```python
import torch
from xlora.xlora_utils import load_model  # type: ignore

XLoRA_model_name = "myuser/repo"

model_loaded, tokenizer = load_model(
    model_name=XLoRA_model_name,
    device="cuda:0",
    dtype=torch.bfloat16,
)
```

### Ölçeklemeleri günlüğe kaydetme (Loglama)
```python
# Enable scalings logging and begin a log
model_created.enable_scalings_logging()

# Run forward passes to accumulate a log

# Write the log to a file, or multiple.
model_created.flush_log_scalings("./path/to/output/file")

# Get a shallow copy of the scalings
log_copy = model_created.get_scalings_log()

# Disable scalings logging
model_created.disable_scalings_logging()

# Clear the scalings log
model_created.clear_scalings_log()

# Get the latest scalings prediction
scalings_pred = model_created.get_latest_scalings()

# Load the scalings log from a file, or multiple automatically.
loaded_log = xlora.xlora_utils.load_scalings_log("./path/to/output/file", verbose=True)
```

### Eğitilebilir parametreler
```python
model: xLoRAModel = ... # Load the model

num_trainable, num_all_params = model.get_nb_trainable_parameters()

model.print_trainable_parameters()
```

### Adaptörlerin eğitilebilirliğini dinamik olarak ayarlamay
```python
model: xLoRAModel = ... # Load the model

# Use trainable adapters: mark all adapters as trainable
model.set_use_trainable_adapters(True)

# Get the current status of the trainable adapters, in this case returning True
model.get_use_trainable_adapters()
```

### Ölçekleme geçiş değerini ayarlama ve sıfırlama
```python
model: xLoRAModel = ... # Load the model

# Set the scaling pass value to 0, meaning that no adapters will contribute to the scaling pass output
model.set_scaling_pass_value(0)

# Allow the model to use the default scaling pass value
model.set_scaling_pass_value(None)
```

### Global LoRA ağırlığını ayarlama ve alma (okuma)
```python
model: xLoRAModel = ... # Load the model

# Multiply the output of each LoRA adapter by 2, additionally to the scalings.
model.set_global_scaling_weight(2)

# Returns 2
res = model.get_global_scaling_weight()
```

### Top-k LoRA değerini ayarlama ve alma
```python
# Use the top 2 lora experts
model_created.set_topk_lora(2)

# Returns 2
res = model_created.get_topk_lora()
```

## API
X-LoRA API'si 3 bölümden oluşur: "Global API", "Model API" ve "Utility API".Genellikle Global API, X-LoRA modelleri oluşturmak için kullanılır; Model API, modellerle etkileşim kurmak için kullanılırken, Utility API ise yararlı yardımcı fonksiyonlar sağlar.

- [Global API](README.md#global-api): `xlora.*`
  - `xlora.add_xlora_to_model`
  - `xlora.from_pretrained`
- [Utility API](README.md#utility-api): `xlora.xlora_utils.*`
  - `xlora.xlora_utils.load_scalings_log`
  - `xlora.xlora_utils.load_model`
- [Model API](README.md#model-api): `xLoraModel.*`
  - [Scalings](README.md#scalings)
    - `xLoraModel.disable_scalings_logging`
    - `xLoraModel.enable_scalings_logging`
    - `xLoraModel.flush_log_scalings`
    - `xLoraModel.get_scalings_log`
    - `xLoraModel.set_scaling_pass_value`
    - `xLoraModel.get_latest_scalings`
    - `xLoraModel.set_global_lora_weight`
    - `xLoraModel.get_global_lora_weight`
  - [Trainable parameters](README.md#trainable-parameters-1)
    - `xLoraModel.get_nb_trainable_parameters`
    - `xLoraModel.print_trainable_parameters`
  - [Trainable adapters](README.md#setting-the-trainable-adapters)
    - `xLoraModel.set_use_trainable_adapters`
    - `xLoraModel.get_use_trainable_adapters`

### X-LoRA Yapılandırması (Config)
X-LoRA Yapılandırması, bir X-LoRA modelinin tam yapılandırmasını kaydeder.
```python
Args:
    hidden_size (`int`):
        Hidden size of the base model.
    device (`torch.device`):
        Device for the X-LoRA classifier.
    enable_softmax (`bool`, *optional*, defaults to `True`):
        Enable softmax application for the X-LoRA classifier.
    enable_softmax_topk (`bool`, *optional*, defaults to `False`):
        Enable softmax application for the top-k LoRA adapters. Mutually exclusive to `enable_softmax` and must only be set if `top_k_lora` is.
    softmax_temperature (`float`, *optional*, defaults to 1.0):
        Softmax temperature, lower yields sharper predictions
    layerwise_scalings (`bool`, *optional*, defaults to `False`):
        Generate scalings for each layer.
    top_k_lora (`int`, *optional*, defaults to None):
        Sparsely select the top_k LoRA experts instead of the default dense method.
    xlora_depth (`int`, *optional*, defaults to 1):
        Depth of the X-LoRA classifier.
    xlora_size (`int`, *optional*, defaults to 2048):
        Hidden size of the X-LoRA classifier, irrelevant if `xlora_depth=1`.
    enable_relu_and_dropout (`bool`, *optional*, defaults to `True`):
        Enable ReLU activation and Dropout application of the X-LoRA classifier.
    use_bias (`bool`, *optional*, defaults to `True`):
        Enable bias in X-LoRA classifier.
    xlora_dropout_p (`float`, *optional*, defaults to 0.2):
        Dropout probability of the X-LoRA classifier, irrelevant if `xlora_depth=1` or `enable_relu_and_dropout=False`.
    stop_token_id (`int`, *optional*):
        The id of the stop token for the input. If this is None, the sequence length is calculated using the attention mask.
    use_trainable_adapters (`bool`, *optional*, defaults to False):
        Make the adapters trainable.
    scaling_pass_value (`float`, *optional*, defaults to 0):
        Scaling pass value.
    global_scaling_weight (`float`, *optional*, defaults to 1):
        Weight to multiply output of each LoRA adapter by.
```

### Global API
- `xlora.add_xlora_to_model(model: PreTrainedModel, xlora_config: xLoRAConfig, adapters: Dict[str, str], verbose: bool) -> xLoraModel`
  - Bir modeli bir xLoraModel'e dönüştürerek sınıflandırıcıyı (classifier) ve adaptörleri örneklendirin (instantiate).
- `xlora.from_pretrained(load_directory: str, model: PreTrainedModel, adapters: adapters: Optional[Dict[str, str]] = None, verbose: bool, device: str, from_safetensors: bool = True) -> xLoraModel`
  - X-LoRA sınıflandırıcısını ve adaptörlerini, belirtilen yerel yoldan veya HuggingFace model kimliğinden (ID) yükleyin. Bu işlem, bir X-LoRA sınıflandırıcısı eğitildikten sonra çağrılmalıdır.

### Utility API
- `xlora.xlora_utils.load_scalings_log(path: str, verbose: bool = False) -> List[torch.Tensor]`
  - İki türü dikkate alarak ölçekleme günlüğünü yükleyin.
- `xlora.xlora_utils.load_model(model_name: str, device: str, dtype: torch.dtype, adapters: Dict[str, str], use_flash_attention_2: bool = False, load_xlora: bool = True, verbose: bool = False) -> Tuple[Union[AutoModelForCausalLM, xLoRAModel], Union[PreTrainedTokenizer, PreTrainedTokenizerFast]`
  - X-LoRA yapılandırmasındaki gibi belirtilen adaptörlerle bir modeli yüklemek ve eğer belirtilirse onu X-LoRA'ya dönüştürmek için kullanılan yardımcı fonksiyon. model_name bir HuggingFace model kimliği (ID) olabilir; bu durumda gerekli tüm ağırlıklar otomatik olarak indirilecektir.

### Model API
#### Scalings
- `xLoraModel.disable_scalings_logging()`
  - Ölçekleme günlüğe kaydetmeyi (scalings logging) durdurun, ancak günlüğü temizlemeyin.
- `xLoraModel.clear_scalings_log()`
  - Ölçekleme günlüğünü temizle.
- `xLoraModel.enable_scalings_logging()`
  - Ölçekleme günlüğe kaydetmeyi (scalings logging) etkinleştirin. Her bir ileri geçiş (forward pass) gerçekleştiğinde, tahmin edilen ölçeklemeler günlüğe kaydedilecektir.
- `xLoraModel.flush_log_scalings(path: str)`
  - Ölçekleme günlüğünü ((num_logged, batch_size, seq_len, n_layers, n_classes) boyutunda bir tensör olarak) belirtilen yola kaydedin.

Eğer tensör oluşturulamazsa, her biri bir dizi uzunluğunu (sequence length) içerecek şekilde, (num_logged, batch_size, seq_len, n_layers, n_classes) boyutunda tensörler içeren birden fazla dosya yazılır. Ayrıca, günlük sırasının yeniden oluşturulabilmesi için, her bir dizi günlük dosyasını içerdiği tensörün indeksine eşleyen bir JSON dosyası çıktı olarak verilir.

Belirtilen dosya (adı) bir uzantı içermemelidir
- `xLoraModel.get_scalings_log(self) -> List[Tensor]`
  - Ölçekleme günlüğünü içeren listenin sığ bir kopyasını (tensörleri değil, sadece listenin kendisini kopyalayarak) döndürür. Listeyi düzenlemek, altta yatan (asıl) günlüğü değiştirmez.
Tensörler `(batch_size, seq_len, n_layers, n_classes)` boyutundadır. `seq_len` boyutu, girdi boyutuna göre değişiklik gösterebilir.
- `xLoraModel.set_scaling_pass_value(self, value: Union[Number, None])`
  - Ölçekleme geçişi sırasında ölçeklemeleri kalıcı olarak belirli bir değere manuel olarak ayarlayın. Varsayılan ölçeklemeleri etkinleştirmek için bu fonksiyonu `None` ile çağırın. Bu değişiklik yapılandırmaya (config) yansıtılır.
- `xLoraModel.get_latest_scalings(self) -> Optional[Tensor]`
  - En son ölçekleme tahminini veya eğer hiç ölçekleme tahmin edilmemişse `None` değerini döndürür. Tensör `(batch_size, seq_len, n_layers, n_classes)` boyutundadır.
- `xLoraModel.set_global_scaling_weight(self, weight: float)`
  - Her bir LoRA adaptörünün çıktısının çarpılacağı bir skaler olan **global LoRA ağırlığını** ayarlayın. Bu değer varsayılan olarak 1'dir ve yapılandırmaya (config) yansıtılır.
- `xLoraModel.get_global_scaling_weight(self) -> float`
  - Global LoRA ağırlığını alın.
#### Eğitilebilir parametreler.
- `xLoraModel.get_nb_trainable_parameters() -> Tuple[int, int]`
  - Desteye dön `(num_trainable, num_all_params)`
- `xLoraModel.print_trainable_parameters()`
  - İşte bu teknik cümlenin Türkçe çevirisi:

Verilen model için, X-LoRA bileşenleri dahil olmak üzere eğitilebilir ve eğitilemez parametreleri yazdırın (gösterin).

#### Eğitilebilir Adaptörlerin Ayarlanması
- `xLoraModel.set_use_trainable_adapters(use_trainable_adapters: bool)`
  - Adaptörlerin eğitilebilirliğini ayarlayın. Bu, yapılandırmaya (config) yansıtılır.
- `xLoraModel.get_use_trainable_adapters(self) -> bool`
  - Adaptörlerin eğitilebilir (trainable) veya eğitilemez (not trainable) durumunu alın.
#### Top-k
- `xLoraModel.set_topk_lora(self, value: Optional[int])`
  - Varsayılan yoğun (dense) yöntem yerine, belirtilen `top_k` LoRA uzmanlarını seyrek (sparse) olarak seçin. Yoğun yöntemi kullanmak için `None` olarak ayarlayın. Bu, yapılandırmaya (config) yansıtılır.
- `xLoraModel.get_topk_lora(self) -> Optional[int]`
  - **Mevcut `top_k` LoRA uzmanları değerini alın.**

## Orijinal makale ve atıf

Bu çalışmaya şu şekilde atıfta bulunun:
```bibtex
@article{Buehler_XLoRA_2024,
    title   = {X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design},
    author  = {E.L. Buehler, M.J. Buehler},
    journal = {},
    year    = {2024},
    volume  = {},
    pages   = {},
    url     = {https://arxiv.org/abs/2402.07148}
}
```

## **Katkıda Bulunma**
Lütfen bir PR (Pull Request) göndermeden önce `make style` komutunu çalıştırın.
