//! Case- and separator-insensitive paths for archive lookup.

/// A path in the form used to index the virtual file system.
///
/// Morrowind's data was authored on Windows: archive entries use `\`, and references to them appear
/// in any mixture of cases. Every lookup key is folded to one form — lowercase ASCII, `/`
/// separators, no leading separator, no repeated separators — so `Meshes\Base_Anim.NIF` and
/// `meshes/base_anim.nif` are the same file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NormalizedPath(String);

impl NormalizedPath {
    /// Folds `raw` into normalized form.
    pub fn new(raw: &str) -> Self {
        let mut out = String::with_capacity(raw.len());
        for c in raw.chars() {
            let c = match c {
                '\\' => '/',
                // ASCII-only, matching the original engine. Unicode case folding is not
                // round-trippable and would make two distinct names collide.
                'A'..='Z' => c.to_ascii_lowercase(),
                _ => c,
            };
            // Collapse repeated separators, and drop a leading one.
            if c == '/' && (out.is_empty() || out.ends_with('/')) {
                continue;
            }
            out.push(c);
        }
        Self(out)
    }

    /// Wraps a string that is already in normalized form.
    ///
    /// Archives normalize their names once while indexing, so re-running [`Self::new`] over them
    /// would rescan every character to no effect. The debug assertion turns the caller's promise
    /// into a checked invariant rather than a comment.
    pub(crate) fn from_normalized(normalized: &str) -> Self {
        debug_assert_eq!(
            Self::new(normalized).as_str(),
            normalized,
            "from_normalized called with a name that is not normalized"
        );
        Self(normalized.to_owned())
    }

    /// The normalized form.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The extension without its dot, if the final component has one.
    pub fn extension(&self) -> Option<&str> {
        let name = self.file_name()?;
        let dot = name.rfind('.')?;
        Some(&name[dot + 1..])
    }

    /// The final path component.
    pub fn file_name(&self) -> Option<&str> {
        if self.0.is_empty() {
            return None;
        }
        Some(match self.0.rfind('/') {
            Some(slash) => &self.0[slash + 1..],
            None => &self.0,
        })
    }

    /// Whether this path is inside `directory`.
    ///
    /// Takes an already-normalized prefix so a caller filtering a whole index normalizes once
    /// rather than once per candidate.
    pub(crate) fn starts_with_directory(&self, directory: &NormalizedPath) -> bool {
        if directory.0.is_empty() {
            return true;
        }
        self.0
            .strip_prefix(&directory.0)
            .is_some_and(|rest| rest.starts_with('/'))
    }
}

impl std::fmt::Display for NormalizedPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_case_and_separators() {
        assert_eq!(
            NormalizedPath::new(r"Meshes\Base_Anim.NIF").as_str(),
            "meshes/base_anim.nif"
        );
        assert_eq!(
            NormalizedPath::new("MESHES/BASE_ANIM.NIF"),
            NormalizedPath::new(r"meshes\base_anim.nif")
        );
    }

    #[test]
    fn drops_leading_and_repeated_separators() {
        assert_eq!(
            NormalizedPath::new("/meshes//x.nif").as_str(),
            "meshes/x.nif"
        );
        assert_eq!(
            NormalizedPath::new(r"\\meshes\\\x.nif").as_str(),
            "meshes/x.nif"
        );
        assert_eq!(NormalizedPath::new("///").as_str(), "");
    }

    #[test]
    fn folds_ascii_only_and_leaves_other_characters_untouched() {
        // Only ASCII is folded, matching the original engine. Unicode folding is not
        // round-trippable, so applying it would let two distinct archive names collide.
        let normalized = NormalizedPath::new("AÄB");
        assert_eq!(normalized.as_str(), "aÄb");
    }

    #[test]
    fn extracts_file_name_and_extension() {
        let path = NormalizedPath::new(r"Textures\Tx_Wood_01.DDS");
        assert_eq!(path.file_name(), Some("tx_wood_01.dds"));
        assert_eq!(path.extension(), Some("dds"));

        let bare = NormalizedPath::new("readme");
        assert_eq!(bare.file_name(), Some("readme"));
        assert_eq!(bare.extension(), None);

        assert_eq!(NormalizedPath::new("").file_name(), None);
        assert_eq!(NormalizedPath::new("").extension(), None);
    }

    #[test]
    fn directory_prefix_matches_whole_components_only() {
        let path = NormalizedPath::new("meshes/f/furn_ex.nif");
        assert!(path.starts_with_directory(&NormalizedPath::new("Meshes")));
        assert!(path.starts_with_directory(&NormalizedPath::new(r"meshes\f")));
        assert!(path.starts_with_directory(&NormalizedPath::new("")));

        // "mesh" is not a component of "meshes/..." and must not match.
        assert!(!path.starts_with_directory(&NormalizedPath::new("mesh")));
        assert!(!path.starts_with_directory(&NormalizedPath::new("textures")));
        // The file itself is not inside itself.
        assert!(!path.starts_with_directory(&NormalizedPath::new("meshes/f/furn_ex.nif")));
    }

    #[test]
    #[should_panic(expected = "not normalized")]
    fn from_normalized_rejects_unnormalized_input_in_debug() {
        let _ = NormalizedPath::from_normalized(r"Meshes\A.NIF");
    }

    #[test]
    fn from_normalized_agrees_with_new_for_already_normalized_input() {
        let normalized = NormalizedPath::new(r"Meshes\A.NIF");
        assert_eq!(
            NormalizedPath::from_normalized(normalized.as_str()),
            normalized
        );
    }
}
