def get_gps_filter(bounds):
    return {"geo_bounding_box": {
        "gps": {
            "top_left": {"lon": float(bounds[0]), "lat": float(bounds[3])},
            "bottom_right": {"lon": float(bounds[2]), "lat": float(bounds[1])}
        }
    }}

def create_time_range_query(start, end, boost=1.0):
    return {
            "range":
            {
                "timestamp":
                {
                    "gt": start,
                    "lt": end,
                    "boost": boost
                }
                }
            }
