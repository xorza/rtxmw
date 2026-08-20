//! `Morrowind.ini`, which is where the game keeps everything the content files do not.

use std::collections::HashMap;

/// One parsed ini file: sections, each a map of key to raw value.
///
/// **The game's own tuning lives here rather than in the ESM** — the moons' sizes and cycles, the
/// star schedule, the length of the day, and the ten weathers with their colours at four times of
/// day. Three of those were read by hand and written into this crate as literals before there was
/// anything to parse the file with; the weathers are too many to do that with.
///
/// Values are kept as they were written. Nothing here knows whether a line is a colour, a count or
/// a duration, so the reader that wants one says which — see [`Self::colour`] and [`Self::number`].
#[derive(Debug, Default)]
pub(crate) struct Ini {
    sections: HashMap<String, HashMap<String, String>>,
}

impl Ini {
    /// Parses the file's text.
    ///
    /// **Nothing here fails.** An ini is a pile of lines the game itself treats as advisory: a
    /// section it does not know is skipped, a key it does not know is ignored, and a line that is
    /// neither is neither. A reader that wants a value it cannot find says so at that point, where
    /// it knows what the value was for — which is a better error than "the file did not parse".
    pub(crate) fn parse(text: &str) -> Self {
        let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut current = String::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(';') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                current = name.to_ascii_lowercase();
                sections.entry(current.clone()).or_default();
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                sections
                    .entry(current.clone())
                    .or_default()
                    .insert(key.trim().to_ascii_lowercase(), value.trim().to_owned());
            }
        }
        Self { sections }
    }

    /// The raw value of `key` in `section`, matched without regard to case.
    ///
    /// The file is inconsistent about it — `Clouds Maximum Percent` beside `Timescale Clouds` — and
    /// a caller quoting the spelling it read in the file should not have to match the file's own
    /// mind about capitals.
    pub(crate) fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.sections
            .get(&section.to_ascii_lowercase())?
            .get(&key.to_ascii_lowercase())
            .map(String::as_str)
    }

    /// The value of `key` read as a number.
    pub(crate) fn number(&self, section: &str, key: &str) -> Option<f32> {
        self.get(section, key)?.parse().ok()
    }

    /// The value of `key` read as a `R,G,B` colour, decoded from sRGB to linear.
    ///
    /// Every colour the game stores was authored against a fixed-function renderer and is
    /// sRGB-encoded — using the bytes directly in a renderer that works in linear light is the same
    /// error as sampling an albedo through a UNORM view. A fourth component is accepted and dropped;
    /// the general `[Weather]` section writes some of its colours with alpha.
    pub(crate) fn colour(&self, section: &str, key: &str) -> Option<glam::Vec3> {
        let raw = self.get(section, key)?;
        let mut parts = raw.split(',').map(|p| p.trim().parse::<u8>().ok());
        let mut next = || parts.next().flatten().map(rtxmw_texture::channel_to_linear);
        Some(glam::Vec3::new(next()?, next()?, next()?))
    }

    /// Every section whose name begins with `prefix`, and the rest of that name.
    ///
    /// What reads the ten weathers: they are `[Weather Clear]` through `[Weather Blizzard]`, and
    /// nothing states the list.
    pub(crate) fn sections_under(&self, prefix: &str) -> Vec<&str> {
        let prefix = prefix.to_ascii_lowercase();
        let mut found: Vec<&str> = self
            .sections
            .keys()
            .filter_map(|name| name.strip_prefix(&prefix))
            .filter(|rest| !rest.is_empty())
            .collect();
        found.sort_unstable();
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_and_keys_are_found_however_they_were_capitalised() {
        let ini = Ini::parse(
            "; a comment\n\
             [Weather Clear]\n\
             Sky Day Color=095,135,203\n\
             Cloud Speed=1.25\n\
             \n\
             [Weather Overcast]\n\
             Sky Day Color=143,146,149\n",
        );
        assert_eq!(ini.get("weather clear", "CLOUD SPEED"), Some("1.25"));
        assert_eq!(ini.number("Weather Clear", "Cloud Speed"), Some(1.25));
        assert_eq!(ini.get("Weather Clear", "no such key"), None);
        assert_eq!(ini.get("No Such Section", "Cloud Speed"), None);

        // **Decoded to linear**, which is the whole reason this knows a colour from a number: 95 of
        // 255 is 0.373 encoded and 0.115 linear, and using the first is every colour in the game
        // coming out too bright.
        let sky = ini.colour("Weather Clear", "Sky Day Color").unwrap();
        assert!((sky.x - 0.1144).abs() < 1e-3, "{sky:?}");
        assert!(sky.z > sky.y && sky.y > sky.x, "a day sky is blue: {sky:?}");

        // And the weathers are found without a list of them written down anywhere.
        assert_eq!(ini.sections_under("weather "), vec!["clear", "overcast"]);
    }

    #[test]
    fn a_line_that_is_neither_a_section_nor_a_pair_is_simply_not_one() {
        // The file has blank lines, comments and a stray word or two, and the game reads it anyway.
        let ini = Ini::parse("nonsense\n[A]\n; note\n\nk = v \nalso nonsense\n");
        assert_eq!(ini.get("a", "k"), Some("v"));
        assert_eq!(ini.get("a", "also nonsense"), None);
    }
}
