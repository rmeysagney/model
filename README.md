# Helio Support Console

Kendi destek verisiyle eğitilmiş, önceden eğitilmiş LLM kullanmayan Flask tabanlı destek öneri arayüzü.

## Yerelde çalıştırma

```bash
pip install -r requirements-training.txt
python app.py
```

Arayüz: `http://127.0.0.1:5050`

## Model eğitimi

```bash
python train_model.py
python model_quality_checks.py
```

Ham eğitim kayıtları ve değerlendirme çıktıları bu repoda izlenmez. Bu dosyalar kişisel veri içerebileceğinden yalnızca güvenli yerel veya özel depolamada tutulmalıdır.

### Etiket grupları

Model, ham verideki yüzlerce ayrıntılı etiketi 19 kararlı destek grubunda
toplar. Böylece öneriler sadece aynı kullanıcı niyetindeki geçmiş yanıtlardan
seçilir. Faturalama özellikle ayrı ele alınır: `Payments & methods`,
`Credits & usage`, `Subscriptions & plans` ve `Pricing, refunds & invoices`.
Diğer gruplar hesap/giriş, cihaz-yedek/veri, rota-planlama, adres-harita-
navigasyon, paylaşım, reklam, güncelleme/uyumluluk, uygulama hataları, erişim,
başlangıç bilgisi, destek iletişimi, özellik isteği ve genel talepleri kapsar.

Gruplar `data_utils.py` içindeki `group_category()` fonksiyonunda tanımlıdır.
Bir grubu bölmeden önce her alt konunun yeterli sayıda eğitim örneği olmasına
dikkat edilir; çok seyrek bir etiketi ayrı sınıfa çevirmek öneri kalitesini
düşürebilir.

## Yayınlama

`render.yaml` Flask uygulamasını Render Web Service olarak başlatır. Canlıda Gunicorn kullanılır ve `deployment_assets/model_english_multilingual.joblib` yayın modeli yüklenir. Bu dosya eğitim sırasında üretilir; ham JSONL kayıtları yayın deposuna eklenmez.
