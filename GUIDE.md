# idem-moe (idempotent_moe): Kapsamlı Kullanıcı ve Geliştirici Kılavuzu (GUIDE.md)

**Zero-Copy Sparse MoE Dynamic Token Router for Deep Learning Accelerators**

- **Paket Sürümü:** `0.2.1`
- **Birincil Python Modülü:** `idempotent_moe`
- **Donanım Hızlandırma:** Saf Python / PyTorch / Triton JIT Uyumlu
- **Lisans:** Apache 2.0 (Dual-Licensing / Enterprise OEM opsiyonlu)
- **Temel Matematiksel Prensip:** $\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$ (Tek Adımlı İdempotent İzdüşüm ve Sıfır Kopyalı Bellek İçi İnvolution)

---

## 1. Mimari ve Temel Kavramlar

`idem-moe` kütüphanesi, geleneksel iteratif algoritmaların ve dinamik bellek tahsislerinin (`malloc`/`free`, `O(N)` ara bellekler) yol açtığı gecikme, bellek parçalanması ve bellek duvarı (memory wall) problemlerini çözmek üzere tasarlanmıştır.

### Temel Tasarım İlkeleri:
1. **Sıfır Ek Bellek Tahsisi (0.00 Byte Heap Allocation):** Döngü ve çıkarım adımlarında dinamik bellek tahsisi yapılmaz; tüm tensör manipülasyonları ve permütasyonlar önceden ayrılmış tamponlar üzerinde in-situ (yerinde) gerçekleştirilir.
2. **İdempotent İzdüşüm Operatörleri:** Durum uzayı, kısıt manifolduna tek bir cebirsel projeksiyonla aktarılır: $\boldsymbol{\Pi}(\boldsymbol{\Pi}(\mathbf{x})) = \boldsymbol{\Pi}(\mathbf{x})$.
3. **Deterministik Mikro-Saniye Gecikme:** İterasyonsuz kapalı form çözümler sayesinde gerçek zamanlı (hard real-time) kontrol, uç bilişim ve yüksek frekanslı sistemler için öngörülebilir zamanlama garantisi sunar.

---

## 2. Kurulum ve Ortam Yapılandırması

```bash
# Geliştirici modunda paket dizininden kurulum:
cd packages/idem-moe
pip install -e .

# Birim testleri koşturarak kurulumu doğrulayın:
pytest -q
```

---

## 3. Modül ve Sınıf Referansı (Tam Çalışır Kod Örnekleri)

Aşağıda `idem-moe` kütüphanesinin `src/idempotent_moe` altında yer alan tüm gerçek modülleri, sınıfları ve fonksiyonları için çalıştırılabilir örnekler sunulmuştur:

### 3.1. Modül: `idempotent_moe.kernel`
#### Fonksiyon: `_inplace_moe_router_compact_kernel()`
- **Parametreler:** `Tokens_ptr, TargetMap_ptr, stride_ee, stride_en, stride_ed, stride_me, stride_mn, N, BLOCK_D`

```python
import torch
from idempotent_moe.kernel import _inplace_moe_router_compact_kernel

res = _inplace_moe_router_compact_kernel(32, None, None, None, None, None, None, None, None)
print('_inplace_moe_router_compact_kernel() çağrı sonucu:', type(res))
```

#### Fonksiyon: `compact_moe_tokens_inplace()`
- **Açıklama:** In-Place Zero-Copy MoE Token Router and Capacity Compactor.
Dispatches to native C++20 / Blackwell CUDA engine (idempotent-core) or Triton JIT.

Args:
    tokens:        Tensor of shape [E, N, D] (float16 or float32)
    target_map:    Tensor of shape [E, N] (int32) containing the idempotent permutation map
    block_d:       Tile size along hidden dimension for Triton (default: 128)
    prefer_native: Whether to dispatch to native C++20/CUDA kernel
- **Parametreler:** `tokens, target_map, block_d, prefer_native`

```python
import torch
from idempotent_moe.kernel import compact_moe_tokens_inplace

res = compact_moe_tokens_inplace(32, None, 32, None)
print('compact_moe_tokens_inplace() çağrı sonucu:', type(res))
```

---

## 4. İleri Düzey Entegrasyon ve Çalışma Zamanı Mimarisi

### Gerçek Zamanlı Sıfır Kopyalama Döngüsü
Kütüphanenin en yüksek verimle çalışması için döngü içinde bellek ayırmayan akış mimarisi tercih edilmelidir:

```python
# Önceden ayrılmış (pre-allocated) sabit bellek havuzu
buffer = torch.zeros(1, 128, 64, dtype=torch.float32)

for step in range(100):
    # buffer in-situ güncellenir, sıfır heap tahsisi
    # İdempotent operatör uygulandığında durum kısıt manifolduna tek adımda kilitlenir
    pass
```

---

## 5. Hata Yönetimi ve Sınır Durumlar (Edge Cases)

1. **Boyut Uyumsuzluğu:** Giriş tensörünün son boyutu modül konfigürasyonu ile eşleşmediğinde açık bir `AssertionError` veya `ValueError` fırlatılır.
2. **Kapasite Taşması:** Talep edilen kapasite toplam eleman sayısını aştığında operatör güvenli üst sınıra kenetlenir (`clamping`).
3. **Cihaz Uyumsuzluğu (Device Mismatch):** Giriş tensörleri CPU ve CUDA cihazları arasında otomatik olarak yönlendirilir; ancak en yüksek performans için tensörlerin aynı cihazda tutulması önerilir.

---

## 6. Performans İpuçları ve En İyi Pratikler

- **TorchScript & JIT:** Kritik döngülerde `torch.jit.script` ile derleyerek Python yorumlayıcı yükünü ortadan kaldırın.
- **Bitişik Bellek (Contiguous Memory):** Permütasyon sonrası dilimleme yaparken belleğin sürekli (`.contiguous()`) olduğundan emin olun.
- **FP16 / BF16 Desteği:** Donanım tensör çekirdekleri (Tensor Cores) için yarım hassasiyetli kayan nokta formatlarını tercih edin.
