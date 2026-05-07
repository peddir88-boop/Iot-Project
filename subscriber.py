"""
Device layer:
Simulates cold-chain sensor data and publishes readings through MQTT.

Realistic behavior:
- Mean-reverting temperature movement for each zone
- Periodic zone transitions: warehouse, transit, delivery
- Fault injection: spike, dropout, duplicate, delay
"""

import json
import logging
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
import config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [SIM] %(message)s")
log = logging.getLogger(__name__)


ZONE_TARGETS = {
    "warehouse": 3.5,
    "transit": 5.0,
    "delivery": 6.5
}

ZONES = list(ZONE_TARGETS.keys())


class Vehicle:
    """Simulates one vehicle sensor with realistic drift and zone changes."""

    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.zone = random.choice(ZONES)
        self.temperature = ZONE_TARGETS[self.zone] + random.gauss(0, 0.5)
        self.humidity = random.uniform(50, 70)
        self.zone_ticks = 0
        self.zone_duration = random.randint(150, 400)

    def update_state(self):
        self.zone_ticks += 1

        if self.zone_ticks >= self.zone_duration:
            self.zone = random.choice(ZONES)
            self.zone_ticks = 0
            self.zone_duration = random.randint(150, 400)

        target_temperature = ZONE_TARGETS[self.zone]

        self.temperature += (
            (target_temperature - self.temperature) * 0.03
            + random.gauss(0, 0.12)
        )

        self.humidity = max(
            30,
            min(90, self.humidity + random.gauss(0, 0.4))
        )

    def reading(self):
        self.update_state()

        return {
            "vehicle_id": self.vehicle_id,
            "zone": self.zone,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": round(self.temperature, 2),
            "humidity": round(self.humidity, 1),
            "fault": None
        }


def inject_fault(reading, fault_probability):
    """
    Randomly mutates a reading to simulate real device faults.

    Returns:
        tuple: (reading, is_dropout)
    """

    if random.random() > fault_probability:
        return reading, False

    fault_type = random.choice([
        "spike",
        "dropout",
        "duplicate",
        "delay"
    ])

    if fault_type == "dropout":
        return None, True

    if fault_type == "spike":
        reading["temperature"] = round(random.uniform(14, 28), 2)
        reading["fault"] = "spike"

    elif fault_type == "duplicate":
        reading["fault"] = "duplicate"

    elif fault_type == "delay":
        time.sleep(random.uniform(1.5, 3.5))
        reading["fault"] = "delay"

    return reading, False


def run():
    cfg = config.load()

    vehicles = [
        Vehicle(vehicle_id)
        for vehicle_id in cfg["simulation"]["vehicles"]
    ]

    topic = cfg["mqtt"]["topic"]
    qos = cfg["mqtt"]["qos"]
    interval = cfg["simulation"]["interval"]
    fault_probability = cfg["simulation"]["fault_probability"]

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(cfg["mqtt"]["host"], cfg["mqtt"]["port"])
    client.loop_start()

    log.info("Simulator started. Publishing to topic: %s", topic)

    published_count = 0

    try:
        while True:
            for vehicle in vehicles:
                reading, dropped = inject_fault(
                    vehicle.reading(),
                    fault_probability
                )

                if dropped:
                    log.debug("Dropout simulated for vehicle: %s", vehicle.vehicle_id)
                    continue

                payload = json.dumps(reading)
                client.publish(topic, payload, qos=qos)
                published_count += 1

                if reading.get("fault") == "duplicate":
                    time.sleep(random.uniform(0.1, 0.8))
                    client.publish(topic, payload, qos=qos)
                    published_count += 1

            if published_count > 0 and published_count % 500 == 0:
                log.info("Published %d messages", published_count)

            time.sleep(interval)

    except KeyboardInterrupt:
        log.info(
            "Simulator stopped. Total messages published: %d",
            published_count
        )

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    run()