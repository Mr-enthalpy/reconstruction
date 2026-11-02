# Hard-negative coverage

There are zero confirmed, typed hard negatives. Existing semantic non-face
acceptance does not establish difficulty. Counts by source and development role
are in `hard_negative_inventory.json`; per-source/ROI details are private.

The legacy CAVE review explicitly excluded three sources for humanoid toy,
sculpture or animal/cartoon face ambiguity. Preserve these as semantic review
items; do not classify them from scene names. The fourth non-admitted source
has a format/annotation gap. These are not verified negatives and have no newly
reviewed ROI. Clear printed human faces stay positive. No new image review or
human adjudication occurred.

UWA's reviewed negatives are clothing/background with possible scale, common
backdrop and illumination shortcuts. They share acquisitions/subjects with
positive ROIs and inherit the same partitions. No difficulty type was supplied
by historical review. CAVE ordinary negatives also need explicit content-based
hardness review. Neither source supports a hard-negative FPR denominator yet.

Mining is permitted on development-train only. Candidate geometry is shared
across B0/B1/capacity control. Validation/final prediction-driven mining requires
a new version and consumes any previously independent test. Future detection
must record missed positives and false proposals, not only successful boxes.
