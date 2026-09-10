"""饮食分析使用的有限知识工具；规则不等于个体医疗建议。"""
RULES = [
    {
        "id": "diversity",
        "text": "日常饮食注重多样性；分析食物种类，不从单餐推断全天营养充足。",
        "source": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    },
    {
        "id": "moderation",
        "text": "关注食盐和高钠调味品的使用；没有用量与营养数据时不能断言低钠达标。",
        "source": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    },
    {
        "id": "evidence",
        "text": "项目证据规则：源食谱每份数据不能直接当作当前家庭成员的摄入量；缺少克重时不计算精确营养目标。",
        "source": "project:evidence-policy",
    },
    {
        "id": "exclusions",
        "text": "项目证据规则：排除列表是硬约束；标签匹配不能证明无过敏原，复合配料和交叉接触情况始终待确认。",
        "source": "project:evidence-policy",
    },
]
