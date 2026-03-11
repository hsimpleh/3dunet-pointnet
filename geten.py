import pkg_resources

# 获取所有已安装库，按名称排序
installed_packages = sorted(
    [(pkg.key, pkg.version) for pkg in pkg_resources.working_set],
    key=lambda x: x[0]
)

# 打印标题
print("="*60)
print("Python 已安装库及版本列表 (sorted by name)")
print("="*60)
# 逐行打印库名和版本，对齐格式更易读
for name, version in installed_packages:
    print(f"{name:<25} | 版本: {version}")
# 打印统计
print("="*60)
print(f"总计安装库数量: {len(installed_packages)}")
print("="*60)

# 单独打印核心库（numpy/torch/tqdm等，方便你检查）
print("\n【核心依赖库单独检查】")
core_libs = ["numpy", "torch", "tqdm", "glob2", "pickle-mixin", "pillow"]
for lib in core_libs:
    try:
        ver = pkg_resources.get_distribution(lib).version
        print(f"{lib:<15} ✅ 版本: {ver}")
    except pkg_resources.DistributionNotFound:
        print(f"{lib:<15} ❌ 未安装")