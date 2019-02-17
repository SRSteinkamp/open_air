# README
Dies hier ist ein Repository zum teil gecloned von:

## Luftqualitaet Daten
Das Ziel dieser Anwendungen ist es die Daten der Stickstoff Sensoren in Köln so auszuwerten, dass diese ähnliche Daten liefern wie die der offiziellen Messstellen.

## Das Problem
1. Niedrige Abdeckung der Stadt Köln mit Sensoren. Viele Sensoren liefern nicht genügend Daten.
2. Daten sind nicht "sauber".
3. Das Regressions Problem an sich ist problematisch. Die Messtellen des LANUVs sind unterschiedlich weit von den verschiedenen Sensoren entfernt. Wir wissen das Werte stark abweichen können.
4. LANUV Sensoren liefern Daten für jede Stunde, höhere Sampling-Rate für Kölner Sensoren (nicht ganz so schlecht)
5. Der eigentliche Sensor für Stickstoff Daten ist nicht der den wir nutzen können, da dieser häufig nicht reliable Daten liefert.
