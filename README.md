# Helio Support Console

Kendi destek verisiyle eğitilmiş, önceden eğitilmiş LLM kullanmayan Flask tabanlı destek öneri arayüzü.

## Yerelde çalıştırma

```bash
pip install -r requirements.txt
python app.py
```

Arayüz: `http://127.0.0.1:5050`

## Model eğitimi

```bash
python train_model.py
python model_quality_checks.py
```

Eğitim kayıtları, değerlendirme çıktıları ve model dosyaları bu repoda izlenmez. Bu dosyalar kişisel veri içerebileceğinden yalnızca güvenli yerel veya özel depolamada tutulmalıdır.

## Yayınlama

Canlı ortama çıkmadan önce kişisel verisi temizlenmiş model paketi ve üretim yapılandırması hazırlanmalıdır. Ayrıntılar için Render veya benzeri bir Python web-service sağlayıcısı kullanılabilir.
