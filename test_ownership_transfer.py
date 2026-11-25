import unittest

from lanc import LanError, compile_source


SCALAR_MOVE = """
    define move_scalar() -> uint8[7,7] {
        define old: uint8 · {storage: heap} = 7;
        define owner: uint8 · {storage: heap} = old;
        define result = owner;
        free(owner);
        return result;
    }
"""

ARRAY_MOVE = """
    define move_array() -> uint8[7,7] {
        define old: uint8[4] · {storage: heap};
        old[2] = 7;
        define owner: uint8[..] · {len: {4}} · {storage: heap} = old;
        define result = owner[2];
        free(owner);
        return result;
    }
"""

RECORD_MOVE = """
    type R = { x: uint8, next: R | null };
    define move_record() -> uint8[7,7] {
        define child: R = { x: 7 };
        define old: R · {storage: heap} = { next: child };
        define owner: R · {storage: heap} = old;
        define result = owner.next.x;
        free(owner);
        return result;
    }
"""

PACKED_MOVE = """
    define move_packed() -> uint8[1,1] {
        define old: bit[8] · {storage: heap} = [1; 8];
        define owner: bit[8] · {storage: heap} = old;
        define result = 0;
        if (owner[0] == 1) { result = 1; }
        free(owner);
        return result;
    }
"""

REFILL = """
    define refill() -> uint8[7,7] {
        define slot: uint8 · {storage: heap} = 1;
        define old_view: uint8 · {pass: alias} = slot;
        free(slot);
        define replacement: uint8 · {storage: heap} = 7;
        slot = replacement;
        define result = slot;
        free(slot);
        return result;
    }
"""


class OwnershipTransferTests(unittest.TestCase):
    def test_moves_preserve_scalar_array_record_and_packed_facts(self):
        c, table = compile_source(SCALAR_MOVE + ARRAY_MOVE + RECORD_MOVE + PACKED_MOVE)
        self.assertEqual(table.count("transfer ownership"), 4)
        self.assertEqual(table.count("moved binding"), 4)
        self.assertRegex(c, r"lan_slice lv_\d+_owner = lv_\d+_old;")
        self.assertRegex(c, r"uint8_t \(\*lv_\d+_owner\)\[1\] = &\(\(\*lv_\d+_old\)\);")

    def test_old_owner_and_preexisting_slice_are_tombstones(self):
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(SCALAR_MOVE.replace("define result = owner;", "define result = old;"))
        with self.assertRaisesRegex(LanError, "already be freed or moved"):
            compile_source(SCALAR_MOVE.replace("free(owner);", "free(old);"))
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(ARRAY_MOVE.replace("old[2] = 7;", "old[2] = 7; define view = old[0,4);")
                           .replace("define result = owner[2];", "define result = view[2];"))
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(RECORD_MOVE.replace("define owner", "define prior: R = { next: old }; define owner")
                           .replace("owner.next.x", "prior.next.x"))

    def test_transfer_requires_a_live_owner_and_disposes_of_the_destination(self):
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(SCALAR_MOVE.replace("free(owner);",
                                               "define twice: uint8 · {storage: heap} = old; free(owner);"))
        with self.assertRaisesRegex(LanError, "heap leak: 'owner'"):
            compile_source(SCALAR_MOVE.replace("free(owner);", ""))
        with self.assertRaisesRegex(LanError, "matching owned array length contract"):
            compile_source(ARRAY_MOVE.replace("{len: {4}}", "{len: {3}}"))

    def test_as_move_is_rejected_with_the_inferred_form(self):
        with self.assertRaisesRegex(LanError, "ownership moves are inferred"):
            compile_source(SCALAR_MOVE.replace("define owner: uint8 · {storage: heap} = old;",
                                               "define owner = old as: move;"))

    def test_explicit_and_inferred_owning_return(self):
        for annotation in ("-> uint8 · {storage: heap}", ""):
            with self.subTest(annotation=annotation):
                _, table = compile_source(f"""
                    define alloc() {annotation} {{
                        define old: uint8 · {{storage: heap}} = 7;
                        return old;
                    }}
                    define f() -> uint8 {{
                        define owner = alloc();
                        define result = owner;
                        free(owner);
                        return result;
                    }}
                """)
                self.assertIn("to result", table)
                self.assertIn("owned result, Alive", table)

    def test_loop_oracle_tracks_move_as_lifetime_write(self):
        _, table = compile_source("""
            define f() -> void {
                define old: uint8 · {storage: heap} = 7;
                define keep_running = true;
                /** @oracle @transition => { state(keep_running) := {0}; state(old) := Tombstone; } */
                while (keep_running) { define owner: uint8 · {storage: heap} = old; free(owner); keep_running = false; }
            }
        """)
        self.assertIn("state(old) := Tombstone;", table)

    def test_refilling_released_slot_uses_new_identity(self):
        _, table = compile_source(REFILL)
        self.assertIn("return {7}", table)
        self.assertRegex(table, r"transfer ownership id#\d+ replacement \[entry\] to id#\d+ slot")
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(REFILL.replace("define result = slot;", "define result = old_view;"))
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(REFILL.replace("define result = slot;", "define result = replacement;"))

    def test_refill_requires_released_destination_on_every_path(self):
        conditional_free = REFILL.replace("define refill()", "define refill(flag: boolean)").replace(
            "free(slot);", "if (flag) { free(slot); }", 1)
        with self.assertRaisesRegex(LanError, "refill requires a released mutable owner"):
            compile_source(conditional_free)
