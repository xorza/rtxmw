//! The `REGN` record's weather table, which is what says a place cannot have snow.

use crate::error::Result;
use crate::esm_reader::Record;

/// How likely each weather is in one region, out of a `REGN` record's `WEAT` subrecord.
///
/// **The one thing in the game that limits weather to a place.** Every exterior cell names a region
/// and every region gives each of the ten a percentage, summing to a hundred — so Vvardenfell's ash
/// wastes are the only place a blight storm blows, and no region on the mainland can snow. A zero
/// is not "rare": it is the game saying that weather does not happen here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionRecord {
    /// The region's own id, as `Cell::region` names it.
    pub id: String,
    /// The chance of each of [`RegionRecord::ORDER`], as a percentage.
    pub chances: [u8; RegionRecord::WEATHERS],
}

impl RegionRecord {
    /// How many weathers a `WEAT` subrecord can carry.
    pub const WEATHERS: usize = 10;

    /// What each byte of `WEAT` is the chance of, which is the order the record stores them in.
    ///
    /// **Not the order anything else lists them in**, which is why it is written down. The ini's
    /// sections come out sorted by name, so a caller matching a chance to a weather has to go
    /// through here rather than by position. Lower-cased to match what an ini section name comes
    /// to.
    pub const ORDER: [&'static str; Self::WEATHERS] = [
        "clear",
        "cloudy",
        "foggy",
        "overcast",
        "rain",
        "thunderstorm",
        "ashstorm",
        "blight",
        "snow",
        "blizzard",
    ];

    /// The eight `WEAT` carried before Bloodmoon added snow and blizzard.
    ///
    /// A record written by the base game stops here, and the two missing are absent rather than
    /// zero — which comes to the same thing and is why a short subrecord is read rather than
    /// refused.
    const BASE_WEATHERS: usize = 8;

    /// Reads the region, or `None` where the record names no weather table.
    pub fn parse(record: &Record<'_>) -> Result<Option<Self>> {
        let (mut id, mut chances) = (None, None);
        for sub in record.subrecords() {
            let sub = sub?;
            match &sub.name().0 {
                b"NAME" => id = Some(sub.as_str().into_owned()),
                b"WEAT" => {
                    let data = sub.data();
                    if data.len() < Self::BASE_WEATHERS {
                        continue;
                    }
                    let mut read = [0u8; Self::WEATHERS];
                    let carried = data.len().min(Self::WEATHERS);
                    read[..carried].copy_from_slice(&data[..carried]);
                    chances = Some(read);
                }
                _ => {}
            }
        }
        Ok(match (id, chances) {
            (Some(id), Some(chances)) => Some(Self { id, chances }),
            _ => None,
        })
    }

    /// Whether `weather` — an ini section name, lower-cased — can occur here.
    ///
    /// Unknown to [`Self::ORDER`] counts as allowed: a weather this table has no column for is one
    /// the record cannot have ruled out.
    pub fn allows(&self, weather: &str) -> bool {
        Self::ORDER
            .iter()
            .position(|named| *named == weather)
            .is_none_or(|column| self.chances[column] > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esm_reader::EsmReader;
    use crate::esm_reader::internals::{SubrecordSpec, push_header, push_record};

    /// A `REGN` record named `id` with the given `WEAT` bytes.
    fn region_esm(id: &str, weather: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_header(&mut bytes);
        let mut name = id.as_bytes().to_vec();
        name.push(0);
        push_record(
            &mut bytes,
            b"REGN",
            0,
            &[
                SubrecordSpec {
                    name: b"NAME",
                    data: &name,
                },
                SubrecordSpec {
                    name: b"WEAT",
                    data: weather,
                },
            ],
        );
        bytes
    }

    fn only(bytes: &[u8]) -> Option<RegionRecord> {
        let reader = EsmReader::new(bytes).expect("should parse");
        let record = reader
            .records()
            .next()
            .expect("one record")
            .expect("should read");
        RegionRecord::parse(&record).expect("should parse")
    }

    #[test]
    fn a_region_rules_out_every_weather_it_gives_no_chance_to() {
        // Ascadian Isles as the game writes it: mostly clear and cloudy, some rain and thunder,
        // and no ash, no blight, no snow at all.
        let bytes = region_esm("Ascadian Isles Region", &[40, 30, 10, 10, 5, 5, 0, 0, 0, 0]);
        let region = only(&bytes).expect("a region with a weather table");
        assert_eq!(region.id, "Ascadian Isles Region");
        assert_eq!(region.chances[..6], [40, 30, 10, 10, 5, 5]);

        for allowed in [
            "clear",
            "cloudy",
            "foggy",
            "overcast",
            "rain",
            "thunderstorm",
        ] {
            assert!(region.allows(allowed), "{allowed} has a chance here");
        }
        for ruled_out in ["ashstorm", "blight", "snow", "blizzard"] {
            assert!(!region.allows(ruled_out), "{ruled_out} is zero here");
        }
        // A name the table has no column for cannot have been ruled out by it.
        assert!(region.allows("tempest"));
    }

    #[test]
    fn a_record_from_before_bloodmoon_carries_eight_and_the_other_two_are_absent() {
        // The base game wrote eight bytes. The two Bloodmoon added read as zero, which is the same
        // answer the file would give: the mainland does not snow.
        let region = only(&region_esm(
            "Bitter Coast Region",
            &[20, 30, 30, 10, 5, 5, 0, 0],
        ))
        .expect("a region with a weather table");
        assert_eq!(region.chances, [20, 30, 30, 10, 5, 5, 0, 0, 0, 0]);
        assert!(region.allows("foggy"));
        assert!(!region.allows("snow") && !region.allows("blizzard"));

        // No table at all is not a region this can answer for.
        assert_eq!(only(&region_esm("Nowhere", &[1, 2, 3])), None);
    }
}
