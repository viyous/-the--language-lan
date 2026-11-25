import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


class UnicodeChineseTests(unittest.TestCase):
    def test_chinese_identifiers_text_records_enums_and_pure_ascii_c(self):
        source = """
            define 最大血量: uint16 = 999;
            define 队伍人数 = 4;
            define 欢迎词 = "你好，Lan世界！";
            define 汉字示例: unicode = '中';

            enum 战斗状态: uint8 {
                溢出 = 254,
                异常 = 255,
            }

            type 角色 = { 血量: uint16[0, 最大血量], 攻击: uint16[0, 500] };
            type 有血量 = interface { 血量: uint16[0, 最大血量] };

            type enum 四维 {
                生命: uint16,
                攻击: uint16,
            }

            define 技能威力: uint8[队伍人数] = [40, 60, 80, 100];

            define 治疗(单位: 有血量, 回复量: uint16) -> void {
                单位.血量 = (单位.血量 + 回复量) as: sat uint16[0, 最大血量];
            }

            define 扣除血量(目标: 角色 · {pass: alias} | null, 伤害: uint16) -> uint16[0, 最大血量] {
                if (目标 == null) {
                    return 0;
                }
                define 剩余: uint16[0, 最大血量] = (目标.血量 - 伤害) as: sat uint16[0, 最大血量];
                目标.血量 = 剩余;
                return 剩余;
            }

            define 解析分数(输入: uint8) -> uint8[0, 100] | 战斗状态 {
                if (输入 > 100) {
                    return 战斗状态.异常;
                }
                return 输入;
            }

            define 累加分数(甲: uint8, 乙: uint8) -> uint8[0, 200] | 战斗状态 {
                define 一 = 解析分数(甲)?;
                define 二 = 解析分数(乙)?;
                return 一 + 二;
            }

            define 读取属性(表: 四维, 键: 四维.key) -> uint16 {
                return 表[键];
            }

            define 求和威力() -> uint16 {
                define 总计: uint16 = 0;
                interation(索引; 0; uint8[0, 队伍人数]; 队伍人数; 1) {
                    总计 += 技能威力[索引];
                }
                return 总计;
            }

            define 文本测试() -> uint32 {
                define 句子: string = "中文测试";
                define 传统串: c_string = "你好" as: c_string;
                define 字符: unicode = '文';
                return (句子.len as uint32) + (传统串.len as uint32) + (欢迎词.len as uint32)
                       + (字符 as uint32) + (汉字示例 as uint32);
            }

            define 拆位(原值: uint32) -> uint16 {
                define 位组: bit[32] = 原值 as: view bit[32];
                define 视图 = 位组[
                    低位: uint16@bit(16, 0, endian: le),
                    高位: uint16@bit(16, 16, endian: le),
                ];
                return (视图.低位 + 视图.高位) as: wrap uint16;
            }

            define 堆测试(初值: uint16[0, 最大血量]) -> uint16[0, 最大血量] {
                define 节点: 角色 · {storage: heap} = { 血量: 初值, 攻击: 10 };
                define 值: uint16[0, 最大血量] = 节点.血量;
                free(节点);
                return 值;
            }

            define main() -> int32 {
                define 主角: 角色 = { 血量: 100, 攻击: 50 };
                治疗(主角, 25);
                define 剩余 = 扣除血量(主角, 30);
                if (剩余 != 95) { return 1; }
                if (求和威力() != 280) { return 2; }
                define 属性: 四维 = { 生命: 300, 攻击: 120 };
                if (读取属性(属性, 四维.key.攻击) != 120) { return 3; }
                if (堆测试(88) != 88) { return 4; }
                if (拆位(0x00030004) != 7) { return 5; }
                if (文本测试() == 0) { return 6; }
                return 0;
            }
        """
        c, _ = compile_source(source)
        # Generated C must be 100% pure ASCII for full C89 / agbcc / cross-compiler portability
        self.assertTrue(c.isascii(), "generated C should be pure ASCII even for Chinese Lan source")
        self.assertIn("int main(void) { return (int)lan_fn_main(); }", c)

    def test_chinese_type_functions_callbacks_and_middle_dot_adjacent(self):
        source = """
            type 定长数组(元素: type, 长度: uint8[1, 16]) {
                return 元素[长度];
            }

            define 递增(数值: uint8) -> uint8 { return (数值 + 1) as: sat uint8; }
            define 递减(数值: uint8) -> uint8 { return (数值 - 1) as: sat uint8; }

            type 步进回调 = callable(递增, 递减);
            type 外部过滤 = extern callable(数值: uint32) -> boolean writes {};

            define 执行(回调: 步进回调, 数据: 定长数组(uint8, 4)·{immutable: true}) -> uint8 {
                return 回调(数据[0]);
            }

            define 过滤调用(过滤器: 外部过滤 | null, 值: uint32) -> boolean {
                if (过滤器 != null) {
                    return 过滤器(值);
                }
                return false;
            }
        """
        c, _ = compile_source(source)
        self.assertTrue(c.isascii())

    def test_chinese_module_export_and_import_lang_and_table(self):
        provider = """
            define 上限: uint16 = 500;
            type 数值 = uint16[0, 上限];
            define 裁剪(输入: uint16) -> 数值 {
                if (输入 > 上限) { return 上限; }
                return 输入;
            }
            export { 上限, 数值, 裁剪 };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch:
            prov = Path(scratch) / "工具.lang"
            prov.write_text(provider, encoding="utf-8")

            # 1. Import from .lang source
            app_lang = """
                import { 上限, 数值, 裁剪 } from "./工具.lang";
                define 测试(值: uint16) -> 数值 {
                    return 裁剪(值);
                }
            """
            app_path = Path(scratch) / "主程序.lang"
            c1, _ = ModuleGraph(app_path, Compiler(app_lang, app_path.name)).compile()
            self.assertTrue(c1.isascii())

            # 2. Export to .table + --export-prefix and import from .table
            c_lib, table = ModuleGraph(
                prov, Compiler(provider, prov.name), library=True, export_prefix="gba_"
            ).compile()
            self.assertTrue(c_lib.isascii())
            self.assertTrue(table.isascii())
            prov.with_suffix(".table").write_text(table, encoding="ascii")

            app_table = """
                import { 上限, 数值, 裁剪 } from "./工具.table";
                define 再测试(值: uint16) -> 数值 {
                    if (值 == 0) { return 上限; }
                    return 裁剪(值);
                }
            """
            c2, _ = compile_source(app_table, Path(scratch) / "调用方.lang")
            self.assertTrue(c2.isascii())

    def test_chinese_alias_field_writes_and_loop_oracle(self):
        source = """
            type 盒子 = { 数值: uint16 };

            define 借用数值(盒: 盒子) -> uint16 · {pass: alias, from: 0, field: 数值} {
                return 盒.数值;
            }

            type 修改回调 = extern callable(盒: 盒子) -> void writes { 盒.数值 };

            define 调用修改(回调: 修改回调, 盒: 盒子) -> uint16 {
                回调(盒);
                return 盒.数值;
            }

            define 倒计时(初值: uint8) -> uint8[0, 0] {
                define 计数: uint8 = 初值;
                /** @oracle
                 * @invariant { state(计数) := uint8[0, 255]; }
                 * @proof { decreases: 计数 }
                 * @ensures { state(计数) := uint8[0, 0]; }
                 */
                while (计数 > 0) {
                    计数 -= 1;
                }
                return 计数;
            }
        """
        c, _ = compile_source(source)
        self.assertTrue(c.isascii())


if __name__ == "__main__":
    unittest.main()
