from skyfield import api
from skyfield import almanac
from datetime import datetime
from datetime import timedelta
import dateutil.parser
from calendar import monthrange

ts = api.load.timescale()
ephem = api.load_file('de421.bsp')

sun = ephem["Sun"]
earth = ephem["Earth"]

# Compute sunrise & sunset for random location near Munich
location = api.Topos('52.1636 N', '4.4802 E', elevation_m=0)
# Compute the sun position as seen from the observer at <location>
sun_pos = (earth + location).at(ts.utc(2024, 4, 15, 12, 0) + 5.0).observe(sun).apparent()
# Compute apparent altitude & azimuth for the sun's position
altitude, azimuth, distance = sun_pos.altaz()

# Print results (example)
print(f"Altitude: {altitude.degrees:.4f} °")
print(f"Azimuth: {azimuth.degrees:.4f} °")