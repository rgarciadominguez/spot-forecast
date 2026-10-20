# 🌊 Spot Forecast

Previsión calibrada de viento y olas para mis spots de windsurf, kite, wing y surf.
No dice "buen día / mal día": da una **horquilla realista (80%)** de lo que te vas a encontrar
y mide, spot a spot, cuánto acierta.

**Dashboard:** https://rgarciadominguez.github.io/spot-forecast/

## Cómo funciona

1. **Previsiones** (Open-Meteo): los mismos modelos que hay detrás de Windguru/Windy
   (ECMWF, GFS, ICON, AROME/ARPEGE, UKMO para viento; ECMWF-WAM, Météo-France, GFS-Wave para olas),
   más cielo, lluvia, temperatura aire/agua, marea y gradiente de presión del Estrecho.
2. **Realidad** (estaciones lo más cerca posible de la orilla):
   - Flecha Extreme (El Portil), cada 5 min.
   - IPMA (Portugal): Sagres, Cabo Raso (Guincho), Cabo Carvoeiro (Peniche), Colares…
   - AEMET (España): Tarifa, Cádiz, costa vasca y cántabra… *(necesita clave gratuita, ver abajo)*
   - METAR de aeropuertos (Gibraltar, Bilbao, Santander, Cascais, Jerez) con histórico para el arranque.
3. **Calibración** (cada hora): por spot y antelación (mismo día, D-1, D-2, D-3, D-5)
   - pesa los modelos según lo que aciertan en ese spot,
   - corrige el sesgo según dirección y fuerza (p.ej. "con levante se queda corto"),
   - construye la horquilla con la distribución real de errores y la abre si los modelos discrepan,
   - mide el acierto con los últimos 30 días que el ajuste no ha visto.
4. **Dashboard** (GitHub Pages): resumen 7 días, detalle hora a hora con previsto vs medido, y fiabilidad.

## Ajustes

- `spots.yaml`: spots (coordenadas, hacia dónde mira la playa, deportes).
- `rules.yaml`: tus rangos aprovechables por deporte (nudos, olas, periodo).

## Clave de AEMET (para Tarifa, Cádiz y Cantábrico)

1. Pide la clave en https://opendata.aemet.es/centrodedescargas/altaUsuario (solo email).
2. En el repo: Settings → Secrets and variables → Actions → New repository secret
   → nombre `AEMET_API_KEY`, valor la clave.

## Comandos

```bash
pip install -r requirements.txt
python -m src.run backfill --start 2025-01-01   # histórico inicial
python -m src.run collect                       # lo que corre cada hora
python -m src.run build                         # solo regenerar dashboard
```

## Próximos pasos

- Boyas de Puertos del Estado / IH Portugal para validar el oleaje.
- Registro de sesiones propias (nota 1-5) para traducir "lo que mide la boya" a "lo que hay en el agua".
- Avisos por Telegram (bueno / muy bueno / excepcional) cuando cada spot supere el 80% de acierto.
