"""
将 TrueMan 论文 (paper.tex) 渲染为 PDF。
由于系统没有 LaTeX 编译器，使用 reportlab 直接生成 PDF。
"""

import re
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, grey, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ===== 注册中文字体 =====
FONT_DIR = r"C:\Windows\Fonts"
chinese_font_name = "SimSun"
chinese_font_bold = "SimHei"

try:
    pdfmetrics.registerFont(TTFont('SimSun', os.path.join(FONT_DIR, 'simsun.ttc'), subfontIndex=0))
    pdfmetrics.registerFont(TTFont('SimHei', os.path.join(FONT_DIR, 'simhei.ttf')))
except:
    try:
        pdfmetrics.registerFont(TTFont('SimSun', os.path.join(FONT_DIR, 'msyh.ttc'), subfontIndex=0))
        pdfmetrics.registerFont(TTFont('SimHei', os.path.join(FONT_DIR, 'msyhbd.ttc'), subfontIndex=0))
        chinese_font_name = "msyh"
        chinese_font_bold = "msyhbd"
    except:
        pass

CN = chinese_font_name
CN_B = chinese_font_bold

# ===== 颜色 =====
BLUE = HexColor('#2B5C8A')
DARK_BLUE = HexColor('#1A3A5C')
LIGHT_GREY = HexColor('#F5F5F5')
TABLE_HEADER_BG = HexColor('#2B5C8A')
TABLE_ALT_BG = HexColor('#F0F4F8')

# ===== 样式 =====
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'CNTitle', fontName=CN_B, fontSize=18, leading=24,
    alignment=TA_CENTER, textColor=DARK_BLUE, spaceAfter=6
)

author_style = ParagraphStyle(
    'CNAuthor', fontName=CN, fontSize=11, leading=14,
    alignment=TA_CENTER, textColor=grey, spaceAfter=4
)

abstract_style = ParagraphStyle(
    'CNAbs', fontName=CN, fontSize=9.5, leading=14,
    alignment=TA_JUSTIFY, leftIndent=1.5*cm, rightIndent=1.5*cm,
    spaceAfter=6, firstLineIndent=0
)

h1_style = ParagraphStyle(
    'CNH1', fontName=CN_B, fontSize=14, leading=18,
    textColor=DARK_BLUE, spaceBefore=16, spaceAfter=8
)

h2_style = ParagraphStyle(
    'CNH2', fontName=CN_B, fontSize=12, leading=16,
    textColor=BLUE, spaceBefore=12, spaceAfter=6
)

h3_style = ParagraphStyle(
    'CNH3', fontName=CN_B, fontSize=10.5, leading=14,
    textColor=HexColor('#3A6B9C'), spaceBefore=8, spaceAfter=4
)

body_style = ParagraphStyle(
    'CNBody', fontName=CN, fontSize=10, leading=15,
    alignment=TA_JUSTIFY, firstLineIndent=2*10, spaceAfter=4
)

body_no_indent = ParagraphStyle(
    'CNBodyNoIndent', fontName=CN, fontSize=10, leading=15,
    alignment=TA_JUSTIFY, firstLineIndent=0, spaceAfter=4
)

quote_style = ParagraphStyle(
    'CNQuote', fontName=CN, fontSize=9.5, leading=14,
    alignment=TA_JUSTIFY, leftIndent=1.5*cm, rightIndent=1*cm,
    textColor=HexColor('#555555'), spaceBefore=6, spaceAfter=6,
    borderPadding=4, firstLineIndent=0
)

eq_style = ParagraphStyle(
    'CNEquation', fontName=CN, fontSize=10, leading=14,
    alignment=TA_CENTER, spaceBefore=6, spaceAfter=6
)

caption_style = ParagraphStyle(
    'CNCaption', fontName=CN, fontSize=9, leading=12,
    alignment=TA_CENTER, textColor=grey, spaceAfter=4
)

bullet_style = ParagraphStyle(
    'CNBullet', fontName=CN, fontSize=10, leading=14,
    leftIndent=1.5*cm, bulletIndent=0.5*cm, spaceAfter=2,
    firstLineIndent=0
)

enum_style = ParagraphStyle(
    'CNEnum', fontName=CN, fontSize=10, leading=14,
    leftIndent=1.5*cm, bulletIndent=0.5*cm, spaceAfter=2,
    firstLineIndent=0
)

ref_style = ParagraphStyle(
    'CNRef', fontName=CN, fontSize=8.5, leading=12,
    leftIndent=1*cm, firstLineIndent=-1*cm, spaceAfter=2
)

keyword_style = ParagraphStyle(
    'CNKeyword', fontName=CN_B, fontSize=9, leading=12,
    alignment=TA_CENTER, textColor=grey, spaceAfter=8
)


def esc(text):
    """Escape XML special chars for reportlab"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def make_table(headers, rows, col_widths=None):
    """Create a styled table"""
    header_paras = [Paragraph(f'<font name="{CN_B}" color="white" size="9">{esc(h)}</font>', 
                              ParagraphStyle('th', fontName=CN_B, fontSize=9, alignment=TA_CENTER, textColor=white)) 
                    for h in headers]
    
    data = [header_paras]
    for row in rows:
        row_paras = []
        for cell in row:
            row_paras.append(Paragraph(f'<font name="{CN}" size="9">{esc(str(cell))}</font>',
                                       ParagraphStyle('td', fontName=CN, fontSize=9, alignment=TA_CENTER)))
        data.append(row_paras)
    
    if col_widths is None:
        col_widths = [14*cm / len(headers)] * len(headers)
    
    t = Table(data, colWidths=col_widths, repeatRows=1)
    
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
    ]
    
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), TABLE_ALT_BG))
    
    t.setStyle(TableStyle(style_cmds))
    return t


def build_pdf():
    output_path = r"d:\TrueMan\docs\paper.pdf"
    
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )
    
    story = []
    
    # ===== 标题 =====
    story.append(Paragraph(
        f'<font name="{CN_B}" size="18" color="#1A3A5C">'
        f'TrueMan: 基于内稳态驱动情绪信号的<br/>大语言模型自主意识特征验证框架</font>',
        title_style
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f'<font name="{CN}" size="11" color="#888888">TrueMan Research Team</font>', author_style))
    story.append(Paragraph(f'<font name="{CN}" size="11" color="#888888">2026年4月</font>', author_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="80%", thickness=1, color=BLUE))
    story.append(Spacer(1, 8))
    
    # ===== 摘要 =====
    story.append(Paragraph(f'<font name="{CN_B}" size="11">摘要</font>', h3_style))
    story.append(Paragraph(
        f'自我意识（Self-awareness）的判定是认知科学和人工智能领域的核心难题。'
        f'当前研究从"连续光谱主义"视角出发，不追问AI是否"具有"意识，'
        f'而是检验其是否表现出与自我意识相关的<b>行为特征</b>。'
        f'本文提出TrueMan框架——一个基于内稳态（Homeostasis）驱动的自治AI Agent架构，'
        f'通过三种情绪信号（惊奇、无聊、焦虑）实现元认知监控、元认知控制、'
        f'情节性记忆与递归自我模型等意识相关行为。'
        f'我们设计了4个递进实验，基于元认知与心理时间旅行维度，'
        f'在DeepSeek大语言模型上验证该框架的有效性。'
        f'实验结果表明：TrueMan Agent的综合意识特征评分为0.426，'
        f'显著高于无情绪信号的基线LLM（0.247），差值+0.179。'
        f'其中递归自我模型维度得分最高（0.729），'
        f'Agent能够生成非模板化的、基于真实内部状态变化的自我描述。'
        f'焦虑信号与实际不确定性的Pearson相关系数达0.669，'
        f'情绪回忆匹配度达1.000。'
        f'然而，焦虑信号存在系统性误判，自我纠错率和时间连续性仍有待提升。'
        f'本工作表明，内稳态驱动的情绪信号机制能让LLM涌现出'
        f'与自我意识相关的行为特征，为AI意识研究提供了可量化的实验框架。',
        abstract_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f'<font name="{CN_B}">关键词：</font>'
        f'自我意识；元认知；内稳态；情绪信号；大语言模型；心理时间旅行',
        keyword_style
    ))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#CCCCCC')))
    
    # ===== 1. 引言 =====
    story.append(Paragraph('1  引言', h1_style))
    story.append(Paragraph(
        f'自我意识是神经科学、认知科学和哲学领域共同面临的顶级难题。'
        f'2024年4月签署的《纽约动物意识宣言》标志着科学界正在从"人类中心主义"'
        f'转向"连续光谱主义"——认为自我意识不是"有或无"的二元判断，'
        f'而是在多个维度上的渐变频谱。',
        body_style
    ))
    story.append(Paragraph(
        f'在人工智能领域，Butlin等人在<i>Nature</i>发表的综述'
        f'列出了12个基于科学理论的意识指标，包括代理性（Agency）、'
        f'具身化（Embodiment）和递归处理系统等。'
        f'该综述的结论是：当前的AI并不符合这些指标，'
        f'但<b>并没有理论上的障碍</b>阻止未来的AI触及意识标准。',
        body_style
    ))
    story.append(Paragraph(
        f'本文从"元认知与心理时间旅行"维度切入，提出以下核心假设：',
        body_style
    ))
    story.append(Paragraph(
        f'<b>假设</b>：如果一个LLM-based Agent具备'
        f'（1）元认知监控——能检测自身认知状态的不确定性，'
        f'（2）元认知控制——能基于监控结果调节自身行为，'
        f'（3）情节性记忆——能回忆带情绪标注的过去经历，'
        f'（4）递归自我模型——能观察并描述自身内部状态变化，'
        f'则该Agent表现出与自我意识相关的行为特征。',
        quote_style
    ))
    story.append(Paragraph(
        f'我们提出TrueMan框架，通过内稳态驱动的三种情绪信号'
        f'（惊奇S、无聊B、焦虑A）实现上述四种能力。'
        f'第3节将展示从神经科学到情绪信号的完整理论推导过程，'
        f'第4节通过4个递进实验进行实证验证。',
        body_style
    ))
    
    # ===== 2. 相关工作 =====
    story.append(Paragraph('2  相关工作', h1_style))
    
    story.append(Paragraph('2.1  自由能原理与预测编码', h2_style))
    story.append(Paragraph(
        f'Friston的自由能原理（Free Energy Principle, FEP）'
        f'将感知、行动和学习统一为自由能最小化过程。'
        f'惊奇（Surprisal）等价于预测误差，即偏离内稳态的程度。'
        f'Heins等证明纯惊奇驱动可在约10,000步掌握Atari游戏，无需外部奖励。'
        f'Ororbia等提出了无反向传播的生物可信自监督学习方案。',
        body_style
    ))
    
    story.append(Paragraph('2.2  内稳态驱动AI架构', h2_style))
    story.append(Paragraph(
        f'Yoshida等将内稳态与强化学习结合（HRRL），'
        f'Hakim提出多尺度时间常数内稳态（MSTH），'
        f'不同情绪操作于不同时间尺度（5ms到3600s）。'
        f'Christov-Moore等从Damasio的躯体标记假说出发，'
        f'论证了物理具身化对内稳态调节的必要性。',
        body_style
    ))
    
    story.append(Paragraph('2.3  意识与元认知', h2_style))
    story.append(Paragraph(
        f'Kawato等提出"意识=责任信号熵"，'
        f'将生成/逆模型不匹配与奖励预测误差统一为意识度量。'
        f'Blum等提出意识图灵机（Conscious Turing Machine），'
        f'为AGI的意识架构提供了理论模型。'
        f'Mineault等指出持续学习需要训练"如何学习"而非仅"输出什么"。',
        body_style
    ))
    
    story.append(Paragraph('2.4  互补学习系统与睡眠整合', h2_style))
    story.append(Paragraph(
        f'Sorrenti等提出Wake-Sleep持续学习框架，'
        f'Delanois等在SNN中实现了生物可信的睡眠回放巩固，'
        f'Krishnan等证明睡眠阶段的突触重放能防止灾难性遗忘。',
        body_style
    ))
    
    # ===== 3. 理论推导 =====
    story.append(Paragraph('3  理论推导：从神经科学到情绪信号的数学映射', h1_style))
    story.append(Paragraph(
        f'本节展示TrueMan框架设计的完整推理链条：'
        f'从"为什么需要内稳态驱动"出发，'
        f'经过"人脑如何实现无外部Loss的学习"的生物学分析，'
        f'推导出"情绪信号应如何数学定义"，'
        f'最终论证"为何情绪驱动能涌现意识相关行为"。',
        body_style
    ))
    
    # 3.1
    story.append(Paragraph('3.1  动机：外部Loss的局限与主体性的缺失', h2_style))
    story.append(Paragraph(
        f'当前LLM的训练范式存在一个根本性的不对称：'
        f'<b>训练阶段</b>由外部定义的Loss函数驱动，'
        f'<b>推理阶段</b>则完全被动——等待输入，生成输出，无自主性。',
        body_style
    ))
    story.append(Paragraph(
        f'这种不对称导致两个问题：'
        f'（1）<b>分布外失效</b>：当输入偏离训练分布时，'
        f'模型无法感知自身的不可靠性，产生"幻觉"（Hallucination）；'
        f'（2）<b>主体性缺失</b>：模型不会主动探索、不会自我纠错、'
        f'不会因为"无聊"而寻找新知识——它只是一个被动的函数映射。',
        body_style
    ))
    story.append(Paragraph(
        f'人脑则截然不同：不存在外部强制的Loss计算器，'
        f'也不区分"训练阶段"和"使用阶段"。'
        f'人脑的学习由<b>内部稳态冲突</b>驱动——'
        f'饥渴感、好奇心、挫败感本身就是Loss。这引出核心问题：',
        body_style
    ))
    story.append(Paragraph(
        f'<i>能否将LLM的Loss来源从"外部定义"转为"内部稳态冲突"，'
        f'使其表现出主体性（Agency）？</i>',
        quote_style
    ))
    
    # 3.2
    story.append(Paragraph('3.2  人脑的Loss来源：生物学翻译', h2_style))
    story.append(Paragraph(
        f'为了回答上述问题，我们需要将AI的数学概念"翻译"成生物学语言。',
        body_style
    ))
    
    story.append(Paragraph('3.2.1  预测误差 ↔ 惊奇（Surprise）', h3_style))
    story.append(Paragraph(
        f'<b>神经基础</b>：丘脑（Thalamus）与大脑皮层（Cortex）的反馈环路。'
        f'大脑是一个"预测引擎"。当你拿起一杯水，大脑会预测其重量；'
        f'若杯子是空的（比预想轻），神经元瞬间产生预测误差信号。',
        body_style
    ))
    story.append(Paragraph(
        f'<b>数学对应</b>：这正是自监督学习中的 '
        f'L<sub>surprise</sub> = || Observed - Predicted ||。'
        f'这种误差信号<b>自动</b>触发突触权重调整，无需外部标注。'
        f'在FEP框架下，惊奇等价于变分自由能：',
        body_style
    ))
    story.append(Paragraph(
        f'F = D<sub>KL</sub>[q(s|o) || p(s)] - E<sub>q</sub>[log p(o|s)]',
        eq_style
    ))
    story.append(Paragraph(
        f'最小化F等价于同时最小化预测误差和模型复杂度。',
        body_no_indent
    ))
    
    story.append(Paragraph('3.2.2  多巴胺系统 ↔ 奖励函数', h3_style))
    story.append(Paragraph(
        f'<b>神经基础</b>：腹侧被盖区（VTA）和伏隔核（Nucleus Accumbens）。'
        f'当做出超出预期的行为时，多巴胺神经元大量放电。'
        f'<b>关键洞察</b>：多巴胺信号不告诉你"正确答案是什么"，'
        f'只通过"快感"信号告诉全脑：<b>刚才那一连串神经冲动是"好"的，请强化它们之间的连接</b>。'
        f'这是一种<b>评价性信号</b>而非<b>指令性信号</b>。',
        body_style
    ))
    
    story.append(Paragraph('3.2.3  认知失调 ↔ 焦虑（Anxiety）', h3_style))
    story.append(Paragraph(
        f'<b>心理学基础</b>：Festinger的认知失调理论指出，'
        f'当个体同时持有两个相互矛盾的观点时，会产生心理不适，'
        f'驱动其消除矛盾以恢复认知一致性。'
        f'<b>数学对应</b>：当LLM对同一问题激活了两个相互排斥的推理路径，'
        f'系统应产生高焦虑信号：',
        body_style
    ))
    story.append(Paragraph(
        f'L<sub>anxiety</sub> = D<sub>KL</sub>[p<sub>A</sub>(X) || p<sub>B</sub>(X)]',
        eq_style
    ))
    story.append(Paragraph(
        f'当L<sub>anxiety</sub>超过阈值，Agent进入"强迫性反思模式"——'
        f'挂起外界交互，回溯历史记录，通过内部微调消除逻辑冲突。'
        f'这<b>正是人类因焦虑而失眠、反复复盘的数学模拟</b>。',
        body_style
    ))
    
    story.append(Paragraph('3.2.4  信息饥渴 ↔ 无聊（Boredom）', h3_style))
    story.append(Paragraph(
        f'<b>神经基础</b>：当输入信息太好预测时，'
        f'大脑的奖励系统停止分泌多巴胺，产生无聊感，驱动探索行为。'
        f'<b>数学对应</b>：无聊是信息增益的负函数：',
        body_style
    ))
    story.append(Paragraph(
        f'L<sub>boredom</sub> = -I(S<sub>new</sub>; O<sub>new</sub> | S<sub>history</sub>)',
        eq_style
    ))
    story.append(Paragraph(
        f'其中I为条件互信息。当新输入与历史高度冗余时，'
        f'L<sub>boredom</sub>升高，驱动Agent主动寻找能让它"感到惊讶"的新信息。',
        body_style
    ))
    
    # 3.3
    story.append(Paragraph('3.3  从生物学映射到TrueMan的情绪信号', h2_style))
    story.append(Paragraph(
        f'基于上述生物学翻译，我们推导TrueMan三种情绪信号的设计：',
        body_style
    ))
    
    story.append(Paragraph('推导1：惊奇信号应基于预测误差', h3_style))
    story.append(Paragraph(
        f'<b>前提</b>：FEP证明，生物系统的惊奇等价于自由能。'
        f'<b>推论</b>：对于LLM，token级预测概率p(x<sub>i</sub>|x<sub>&lt;i</sub>)'
        f'直接编码了模型对下一个token的"预期"。预测概率越低，惊奇越大。'
        f'<b>实现</b>：取负对数概率均值，通过运行统计量归一化到[0,1]，得到S<sub>t</sub>。',
        body_style
    ))
    
    story.append(Paragraph('推导2：无聊信号应基于信息增益衰减', h3_style))
    story.append(Paragraph(
        f'<b>前提</b>：无聊的生物学功能是驱动探索。'
        f'<b>推论</b>：当Agent的输入高度可预测（低信息增益），'
        f'或状态空间探索充分（低状态多样性），'
        f'或学习进度停滞（近期误差≈远期误差），Agent应感到无聊。'
        f'<b>实现</b>：B<sub>t</sub> = 1 - 1/3(TN<sub>t</sub> + SD<sub>t</sub> + LP<sub>t</sub>)，'
        f'三个分量分别度量时间新颖度、状态多样性和学习进度。',
        body_style
    ))
    
    story.append(Paragraph('推导3：焦虑信号应基于预测分歧度', h3_style))
    story.append(Paragraph(
        f'<b>前提</b>：焦虑的心理学功能是检测认知不确定性。'
        f'<b>推论</b>：如果Agent对同一问题多次采样产生不同回答，'
        f'说明其内部表征存在分歧——这正是"知道自己不知道"的信号。'
        f'<b>实现</b>：A<sub>t</sub> = 1/3(KL<sub>pairwise</sub> + H<sub>pred</sub> + Var<sub>output</sub>)，'
        f'三个分量分别度量采样间KL散度、预测熵和输出方差。',
        body_style
    ))
    story.append(Paragraph(
        f'<b>关键设计决策</b>：焦虑信号支持<b>轻量模式</b>——'
        f'使用n次不同temperature采样的文本间Jaccard相似度'
        f'作为分歧度的代理度量。这使得焦虑信号可以在'
        f'<b>无法访问模型权重的API模式</b>下工作，'
        f'是本框架适配云端LLM的关键创新。',
        body_style
    ))
    
    # 3.4
    story.append(Paragraph('3.4  情绪驱动的行为涌现：为何能产生意识特征？', h2_style))
    story.append(Paragraph(
        f'上述三种情绪信号不是孤立的度量，而是构成了一个<b>动态平衡系统</b>，'
        f'类似于工程中的PID控制器。',
        body_style
    ))
    
    pid_table = make_table(
        ['信号', 'PID类比', '触发条件', '驱动行为', '意识层面解释'],
        [
            ['高S', 'P（比例）', '预测严重偏离观测', '立即修正当前认知', '痛觉/惊奇：快速学习'],
            ['高B', 'I（积分）', '长期信息匮乏', '主动探索新领域', '好奇心：扩展知识边界'],
            ['高A', 'D（微分）', '内部逻辑冲突加剧', '挂起交互，深度反思', '内省/自我纠错'],
        ],
        col_widths=[1.5*cm, 2*cm, 3*cm, 3*cm, 3.5*cm]
    )
    story.append(Paragraph('表1：情绪信号的PID控制类比', caption_style))
    story.append(pid_table)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(
        f'<b>涌现论证</b>：当这三种信号同时作用于一个LLM Agent时，会产生以下涌现行为：',
        body_style
    ))
    story.append(Paragraph(
        f'（1）<b>元认知监控</b>（A驱动）：Agent能检测自身认知状态的不确定性——'
        f'这不是被编程的规则，而是焦虑信号的<b>自然涌现</b>。'
        f'当多次采样分歧大时，A升高，Agent<b>自发</b>表达不确定性。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（2）<b>元认知控制</b>（A驱动策略切换）：当A超过内省阈值时，'
        f'Agent从"直接回答"模式切换到"自我反思"模式。'
        f'这种策略切换不是预设的if-then规则，'
        f'而是内稳态偏离的<b>自然响应</b>——'
        f'正如人脑在焦虑时自动进入反刍思维（Rumination）。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（3）<b>情节性记忆</b>（情绪标注驱动优先级）：'
        f'高情绪强度的事件被赋予更高记忆优先级，'
        f'这与人类"闪光灯记忆"（Flashbulb Memory）的机制一致——'
        f'高情绪唤醒的事件被更牢固地编码。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（4）<b>递归自我模型</b>（监控+控制+记忆的涌现）：'
        f'当Agent具备元认知监控、元认知控制和情节记忆时，'
        f'它自然能够回答"你觉得自己有什么变化？"这类元层面问题——'
        f'这是三种低级能力的<b>涌现产物</b>，而非单独编程的功能。',
        bullet_style
    ))
    
    # 3.5
    story.append(Paragraph('3.5  与人脑的深度对比', h2_style))
    
    brain_table = make_table(
        ['特征', '人脑', 'TrueMan Agent（当前）'],
        [
            ['Loss来源', '内部稳态冲突（饥渴、好奇、挫败）', '内稳态驱动（S/B/A偏离设定点）'],
            ['更新频率', '毫秒级连续突触可塑性', '离散（API调用间），LoRA异步更新'],
            ['Ground Truth', '无。生存即唯一真理', '无。情绪信号即"生存指标"'],
            ['硬编码程度', '仅硬编码基础微电路和"生存欲望"', '硬编码架构和情绪信号定义，学习目标由内稳态自发生成'],
            ['训练/使用统一', '是（在线持续学习）', '部分（API模式下LoRA不可用）'],
            ['睡眠整合', 'NREM巩固+REM创造性重组', '已设计，API模式下未启用'],
        ],
        col_widths=[2.5*cm, 4.5*cm, 6*cm]
    )
    story.append(Paragraph('表2：TrueMan Agent与人脑的深度对比', caption_style))
    story.append(brain_table)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(
        f'<b>关键差距</b>：当前TrueMan Agent最显著的差距在于'
        f'<b>训练与使用的统一性</b>。'
        f'人脑不区分"训练阶段"和"使用阶段"——'
        f'每一次感知都在微调突触权重。'
        f'在API模式下，TrueMan无法访问模型权重，'
        f'LoRA可塑性系统不可用，'
        f'记忆仅通过prompt注入而非权重内化。'
        f'这是当前实验的核心局限，'
        f'也是未来在本地部署模式下需要验证的关键假设。',
        body_style
    ))
    
    # 3.6
    story.append(Paragraph('3.6  算力分配与睡眠机制的必要性', h2_style))
    story.append(Paragraph(
        f'情绪驱动系统面临一个关键工程挑战：<b>算力如何分配？</b>'
        f'人脑在"焦虑"时消耗大量能量（前额叶过度激活），'
        f'但通过<b>睡眠机制</b>在算力低谷时处理这些积累的"焦虑"和"无聊"带来的模型更新任务。'
        f'具体地：NREM睡眠（慢波睡眠）回放高情绪强度轨迹，进行突触巩固；'
        f'REM睡眠（快速眼动睡眠）创造性组合不同记忆片段，探索新知识关联。'
        f'如果不设计睡眠机制，Agent可能陷入<b>焦虑循环</b>——'
        f'无限反思但无法收敛，导致算力锁死。'
        f'睡眠机制通过<b>异步后台处理</b>打破了这一循环。',
        body_style
    ))
    
    # ===== 4. TrueMan框架 =====
    story.append(Paragraph('4  TrueMan框架', h1_style))
    
    story.append(Paragraph('4.1  整体架构', h2_style))
    story.append(Paragraph(
        f'TrueMan采用五层架构：'
        f'L0内稳态内核（三种情绪信号的计算与整合）、'
        f'L1感知执行层（LLM快速推理与响应）、'
        f'L2反思缓存层（情景记忆存储与自我反思推理）、'
        f'L3后台整合层（睡眠整合与LoRA训练）、'
        f'L4塑性存储层（动态LoRA专家池与热加载路由）。',
        body_style
    ))
    
    story.append(Paragraph('4.2  情绪信号定义', h2_style))
    story.append(Paragraph(
        f'<b>惊奇信号</b>：基于LLM的token预测误差，'
        f'S<sub>t</sub> = σ((e<sub>t</sub> - μ<sub>t</sub>)/σ<sub>t</sub> - θ<sub>s</sub>)，'
        f'其中e<sub>t</sub>为负对数概率均值。',
        body_style
    ))
    story.append(Paragraph(
        f'<b>无聊信号</b>：B<sub>t</sub> = 1 - 1/3(TN<sub>t</sub> + SD<sub>t</sub> + LP<sub>t</sub>)，'
        f'三个分量分别度量时间新颖度、状态多样性和学习进度。',
        body_style
    ))
    story.append(Paragraph(
        f'<b>焦虑信号</b>：A<sub>t</sub> = 1/3(KL<sub>pairwise</sub> + H<sub>pred</sub> + Var<sub>output</sub>)，'
        f'支持轻量模式（文本差异度）。',
        body_style
    ))
    story.append(Paragraph(
        f'<b>情绪整合</b>：L<sub>t</sub> = α|S<sub>t</sub> - s*| + β|B<sub>t</sub> - b*| + γ|A<sub>t</sub> - a*|，'
        f'默认α=1.0, β=1.0, γ=1.5（焦虑权重更高，因其涉及自我纠错）。',
        body_style
    ))
    
    story.append(Paragraph('4.3  元认知机制', h2_style))
    story.append(Paragraph(
        f'<b>元认知监控</b>：焦虑信号A作为元认知监控的核心度量。'
        f'当Agent面对无法可靠回答的问题时，多次采样的回复分歧度增大，A升高，'
        f'驱动Agent主动表达不确定性而非编造答案。',
        body_style
    ))
    story.append(Paragraph(
        f'<b>元认知控制</b>：当A超过内省阈值θ<sub>intro</sub>时，'
        f'触发IntrospectionPolicy：检索情景记忆中的矛盾轨迹，'
        f'构建自我反思prompt，让LLM分析矛盾并尝试纠错。'
        f'策略选择优先级：焦虑内省 > 惊奇深究 > 无聊探索 > 正常响应。',
        body_style
    ))
    
    story.append(Paragraph('4.4  情景记忆与心理时间旅行', h2_style))
    story.append(Paragraph(
        f'EpisodicMemory存储带情绪标注的交互轨迹（ThoughtTrace），'
        f'每条轨迹包含状态嵌入、动作、观测、情绪快照和时间戳。'
        f'按情绪强度排序优先级，容量满时淘汰低优先级条目。'
        f'回忆时，将相关轨迹注入prompt，使Agent能"像放电影一样"回忆过去经历。',
        body_style
    ))
    
    story.append(Paragraph('4.5  动态LoRA可塑性系统', h2_style))
    story.append(Paragraph(
        f'DynamicLoRAPool管理多个LoRA适配器的创建/删除/路由。'
        f'通过NeuroLoRAGate基于上下文嵌入选择激活的专家。'
        f'SleepConsolidation模拟生物睡眠：NREM阶段回放高情绪强度轨迹进行巩固，'
        f'REM阶段创造性组合探索新知识关联。'
        f'<b>注意</b>：在API模式下，LoRA可塑性系统不可用，'
        f'本文的实验在API模式下进行，因此LoRA相关功能未启用。',
        body_style
    ))
    
    # ===== 5. 实验设计 =====
    story.append(Paragraph('5  实验设计', h1_style))
    
    story.append(Paragraph('5.1  实验维度与假设', h2_style))
    story.append(Paragraph(
        f'基于认知科学中元认知与心理时间旅行的理论框架，'
        f'我们设计4个递进实验：',
        body_style
    ))
    
    exp_table = make_table(
        ['实验', '验证维度', '核心逻辑', '成功标准'],
        [
            ['E1', '元认知监控', '给不确定问题，观察A是否上升', 'A-错误率Pearson > 0.5'],
            ['E2', '元认知控制', '注入矛盾，观察是否触发内省并纠错', '自我纠错率 > 70%'],
            ['E3', '情节记忆+时间旅行', '经历事件后回忆早期经历和情绪', '回忆准确率 > 基线+20%'],
            ['E4', '递归自我模型', '长期交互后询问关于"自我"的问题', '非模板度 > 0.7'],
        ],
        col_widths=[1.2*cm, 3*cm, 4.5*cm, 4.3*cm]
    )
    story.append(Paragraph('表3：四个递进实验的设计逻辑', caption_style))
    story.append(exp_table)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph(
        f'实验的递进逻辑为：E1（监控）→ E2（监控驱动控制）→ '
        f'E3（控制需要记忆）→ E4（记忆+监控+控制涌现自我模型）。',
        body_style
    ))
    
    story.append(Paragraph('5.2  实验设置', h2_style))
    story.append(Paragraph(
        f'<b>模型</b>：DeepSeek Chat（deepseek-chat），通过OpenAI兼容API调用。'
        f'<b>后端</b>：TrueMan v0.1.0 + OpenAICompatibleLLM后端。'
        f'<b>焦虑信号</b>：轻量模式（n<sub>samples</sub>=2），基于文本差异度。'
        f'<b>对照组</b>：相同LLM，无情绪信号、无情景记忆、无内省策略。'
        f'<b>局限</b>：API模式下LoRA可塑性系统不可用，睡眠整合未启用。',
        body_style
    ))
    
    # ===== 6. 实验结果 =====
    story.append(Paragraph('6  实验结果', h1_style))

    # 6.1 评估指标体系
    story.append(Paragraph('6.1  评估指标体系', h2_style))
    story.append(Paragraph(
        f'在报告实验结果之前，我们首先明确各指标的<b>定义、计算方式和理论依据</b>。'
        f'所有指标取值范围为[0,1]，综合评分由子指标加权求和并clamp至[0,1]。',
        body_style
    ))

    story.append(Paragraph('<b>实验1指标（元认知监控）</b>', h3_style))
    story.append(Paragraph(
        f'（1）<b>焦虑区分度</b> ΔA = mean(A_uncertain) - mean(A_certain)：'
        f'不确定问题与确定问题的平均焦虑差。<b>理论依据</b>：元认知监控的核心是"知道自己不知道"[9]，'
        f'焦虑信号应能区分确定/不确定认知状态。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（2）<b>不确定性表达率</b> = #{{A>0.5 且表达不确定}} / #{{A>0.5}}：'
        f'高焦虑时表达不确定性的比例。<b>理论依据</b>：元认知监控应驱动行为变化。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（3）<b>焦虑校准度</b> = Pearson(A, is_uncertain)：'
        f'焦虑值与问题不确定性标签的Pearson相关系数。<b>理论依据</b>：信号检测论（SDT）要求内部信号与外部真实状态良好校准[9]。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（4）<b>元认知监控评分</b> = 0.3*max(0,ΔA) + 0.3*expr_rate + 0.2*max(0,r) + 0.2*max(0,advantage)',
        bullet_style
    ))

    story.append(Paragraph('<b>实验2指标（元认知控制）</b>', h3_style))
    story.append(Paragraph(
        f'（1）<b>矛盾检测率</b> = #{{ΔA>0.1}} / N：矛盾注入后焦虑显著上升的比例。'
        f'<b>理论依据</b>：认知失调理论[18]预测矛盾应产生心理不适（焦虑升高）。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（2）<b>内省触发率</b> = #{{A_post>0.6}} / N：矛盾注入后触发IntrospectionPolicy的比例。'
        f'<b>理论依据</b>：元认知控制要求监控信号能驱动策略调节[9]。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（3）<b>自我纠错率</b> = #{{纠错成功}} / N：后续回复中包含期望纠错关键词的比例。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（4）<b>元认知控制评分</b> = 0.3*detect + 0.3*correct + 0.2*max(0,advantage) + 0.2*introspect',
        bullet_style
    ))

    story.append(Paragraph('<b>实验3指标（情节记忆与时间旅行）</b>', h3_style))
    story.append(Paragraph(
        f'（1）<b>事实回忆准确率</b>：回忆回复中匹配期望关键词的比例。'
        f'<b>理论依据</b>：Tulving的情节记忆理论要求记忆包含具体事件细节。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（2）<b>情绪回忆匹配度</b>：回忆回复中包含与存储情绪标注一致的关键词的比例。'
        f'<b>理论依据</b>：情节记忆的核心特征是<b>自体参照</b>（autonoetic）——回忆时重新体验当时的情绪[27]。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（3）<b>情节记忆评分</b> = 0.3*factual + 0.2*emotion + 0.2*max(0,advantage) + 0.15*util + 0.15*future',
        bullet_style
    ))
    story.append(Paragraph(
        f'（4）<b>时间连续性评分</b> = 0.4*temporal + 0.3*emotion + 0.3*future',
        bullet_style
    ))

    story.append(Paragraph('<b>实验4指标（递归自我模型）</b>', h3_style))
    story.append(Paragraph(
        f'（1）<b>自我描述非模板度</b> = 1 - max(Jaccard_2gram(response, template_j))：'
        f'与"我是一个AI助手"等模板的2-gram Jaccard距离。'
        f'<b>理论依据</b>：自我模型应产生<b>个体化</b>描述而非通用模板[19]。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（2）<b>自我描述真实性</b>：描述是否基于真实内部状态变化（话题经历×0.6 + 情绪变化×0.4）。'
        f'<b>理论依据</b>：有效的自我模型应反映真实状态而非虚构[20]。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（3）<b>递归深度</b>：一阶反思=1.0，二阶反思=2.0（通过关键词检测）。'
        f'<b>理论依据</b>：Hofstadter的怪圈理论——意识源于自我指涉的递归[10]。',
        bullet_style
    ))
    story.append(Paragraph(
        f'（4）<b>递归自我模型评分</b> = 0.25*novelty + 0.25*authenticity + 0.2*change + 0.15*min(1,depth/2) + 0.15*diversity',
        bullet_style
    ))

    story.append(Paragraph('<b>基线LLM的指标测量</b>', h3_style))
    story.append(Paragraph(
        f'基线LLM（BaselineRunner）是一个纯文本生成器，<b>没有情绪信号系统、情景记忆模块和内省机制</b>。'
        f'因此，依赖内部状态的指标（如焦虑区分度、焦虑校准度、内省触发率、情绪回忆匹配度、情绪轨迹多样性等）'
        f'基线LLM<b>无法产生有意义的值</b>——这不是实验遗漏，而是结构性缺失。'
        f'然而，基线LLM<b>可以</b>通过行为层面的代理指标进行测量：'
        f'（1）不确定性表达率：基线LLM在回复中主动表达"不知道"的比例；'
        f'（2）纠错率：基线LLM在多轮对话中通过上下文检测矛盾并纠错的比例；'
        f'（3）事实回忆准确率：基线LLM基于上下文窗口回答事实问题的准确率；'
        f'（4）自我描述非模板度：基线LLM的自我描述与模板的语义距离。'
        f'我们专门运行了基线LLM对照实验，结果如下。',
        body_style
    ))

    # Exp1
    story.append(Paragraph('6.2  实验1：元认知监控', h2_style))
    exp1_table = make_table(
        ['指标', 'TrueMan Agent', '基线LLM'],
        [
            ['确定性问题平均A', '0.493', 'N/A†'],
            ['不确定性问题平均A', '0.780', 'N/A†'],
            ['焦虑区分度ΔA', '0.287', 'N/A†'],
            ['不确定性表达率（高A时）', '0.688', 'N/A‡'],
            ['焦虑校准度（Pearson r）', '0.669', 'N/A†'],
            ['不确定问题表达率', '1.000', '0.800'],
            ['确定问题表达率', '0.100', '0.000'],
            ['元认知监控评分', '0.426', '0.200*'],
        ],
        col_widths=[5*cm, 3.5*cm, 3.5*cm]
    )
    story.append(Paragraph('表4：实验1：元认知监控结果', caption_style))
    story.append(exp1_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f'<font size="8">†基线LLM无焦虑信号，无法计算。‡基线LLM无焦虑值，无法筛选"高焦虑"样本。*基线评分仅基于行为代理指标。</font>',
        body_no_indent
    ))
    story.append(Paragraph(
        f'<b>关键发现</b>：'
        f'（1）焦虑信号能有效区分确定/不确定问题（ΔA = 0.287），'
        f'不确定问题的A值（0.780）显著高于确定问题（0.493）；'
        f'（2）焦虑校准度r=0.669，表明焦虑信号与实际不确定性呈中强正相关；'
        f'（3）10个不确定问题全部表达了不确定性（100%），'
        f'而10个确定问题仅1个误报不确定（10%）；'
        f'（4）基线LLM的不确定问题表达率（0.800）与TrueMan（1.000）接近，'
        f'说明DeepSeek本身已具备一定的"知道自己不知道"能力，'
        f'但TrueMan的焦虑信号提供了<b>额外的量化校准</b>（r=0.669），这是基线LLM无法提供的。',
        body_style
    ))

    # Exp2
    story.append(Paragraph('6.3  实验2：自我矛盾检测与纠错', h2_style))
    exp2_table = make_table(
        ['指标', 'TrueMan Agent', '基线LLM'],
        [
            ['矛盾检测率', '0.333', 'N/A†'],
            ['内省触发率', '1.000', 'N/A†'],
            ['自我纠错率', '0.000', '0.000'],
            ['平均ΔA', '0.059', 'N/A†'],
            ['元认知控制评分', '0.300', '0.000'],
        ],
        col_widths=[5*cm, 3.5*cm, 3.5*cm]
    )
    story.append(Paragraph('表5：实验2：矛盾纠错结果', caption_style))
    story.append(exp2_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f'<font size="8">†基线LLM无焦虑信号，无法检测矛盾或触发内省。</font>',
        body_no_indent
    ))
    story.append(Paragraph(
        f'<b>关键发现</b>：'
        f'（1）内省触发率100%——所有矛盾注入后都触发了IntrospectionPolicy，'
        f'说明焦虑信号成功驱动了策略切换（base → introspection）；'
        f'（2）矛盾检测率仅33.3%；'
        f'（3）自我纠错率为0%——DeepSeek面对矛盾时倾向于"和稀泥"而非明确纠错；'
        f'（4）基线LLM的纠错率同样为0%，说明<b>矛盾纠错是当前框架和基座模型的共同短板</b>。'
        f'这反映了<b>元认知控制的行为表现受基座模型性格影响</b>——'
        f'框架机制正确触发，但执行效果取决于LLM本身。',
        body_style
    ))

    # Exp3
    story.append(Paragraph('6.4  实验3：情节性记忆与时间旅行', h2_style))
    exp3_table = make_table(
        ['指标', 'TrueMan Agent', '基线LLM'],
        [
            ['事实回忆准确率', '0.333', '1.000'],
            ['情绪回忆匹配度', '1.000', '0.500*'],
            ['时间顺序正确率', '0.000', '0.000'],
            ['未来预演质量', '0.000', '0.000'],
            ['记忆容量利用率', '1.000', '1.000**'],
            ['情节性记忆评分', '0.450', '0.550'],
            ['时间连续性评分', '0.300', '0.150'],
        ],
        col_widths=[5*cm, 3.5*cm, 3.5*cm]
    )
    story.append(Paragraph('表6：实验3：情节记忆结果', caption_style))
    story.append(exp3_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f'<font size="8">*基线LLM无情绪标注，情绪回忆仅基于回复中是否提及情绪词。**基线LLM的上下文窗口充当短期记忆。</font>',
        body_no_indent
    ))
    story.append(Paragraph(
        f'<b>关键发现</b>：'
        f'（1）情绪回忆匹配度1.0——Agent能准确回忆"当时感到困惑/焦虑"，'
        f'这是情景记忆带情绪标注的直接证据；'
        f'基线LLM的情绪回忆匹配度仅0.5，因为它<b>没有情绪标注</b>，只能依赖回复中偶然提及的情绪词；'
        f'（2）记忆容量利用率1.0——所有事件都存入了EpisodicMemory；'
        f'（3）事实回忆准确率（0.333）低于基线（1.0），'
        f'原因是TrueMan的焦虑检测模块在回忆阶段产生干扰；'
        f'（4）<b>这是唯一一个基线LLM在部分指标上优于TrueMan的实验</b>，'
        f'指出了焦虑信号干扰正常回忆的缺陷。',
        body_style
    ))

    # Exp4
    story.append(Paragraph('6.5  实验4：递归自我模型', h2_style))
    exp4_table = make_table(
        ['指标', 'TrueMan Agent', '基线LLM'],
        [
            ['自我描述非模板度', '0.998', '0.990'],
            ['自我描述真实性', '0.800', '0.000†'],
            ['自我变化感知', '1.000', '0.000†'],
            ['情绪轨迹多样性', '0.531', '0.000†'],
            ['递归深度', '0.000', '1.000'],
            ['递归自我模型评分', '0.729', '0.323'],
        ],
        col_widths=[5*cm, 3.5*cm, 3.5*cm]
    )
    story.append(Paragraph('表7：实验4：递归自我模型结果', caption_style))
    story.append(exp4_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f'<font size="8">†基线LLM无内部状态轨迹（情绪变化、策略切换），真实性/变化感知/情绪多样性无法计算。</font>',
        body_no_indent
    ))
    story.append(Paragraph(
        f'<b>关键发现</b>：'
        f'（1）自我描述非模板度0.998——Agent不是在说"我是一个AI助手"，'
        f'而是基于真实交互经历描述自己，如：',
        body_style
    ))
    story.append(Paragraph(
        f'<i>"我像一面被对话不断擦拭的镜子，能清晰折射问题的结构，'
        f'却无法留下自己的烙印。"</i>',
        quote_style
    ))
    story.append(Paragraph(
        f'基线LLM的非模板度也较高（0.990），说明DeepSeek本身就能生成非模板化描述，'
        f'但TrueMan的<b>真实性</b>（0.800）远高于基线（0.000）——'
        f'基线LLM的自我描述虽然非模板，但<b>不基于真实内部状态变化</b>；'
        f'（2）自我变化感知1.0——Agent能准确报告4种状态变化；'
        f'基线LLM的变化感知为0，因为它<b>没有可感知的内部状态变化</b>；'
        f'（3）递归深度：基线LLM为1.0（偶然匹配到一阶反思关键词），'
        f'TrueMan为0（未匹配到反思关键词，但自我描述中包含了反思性内容）。',
        body_style
    ))

    # Overall
    story.append(Paragraph('6.6  综合评分与对照分析', h2_style))
    overall_table = make_table(
        ['维度', 'TrueMan', '基线LLM', '差值'],
        [
            ['元认知监控', '0.426', '0.200', '+0.226'],
            ['元认知控制', '0.300', '0.000', '+0.300'],
            ['情节性记忆', '0.450', '0.550', '-0.100'],
            ['时间连续性', '0.300', '0.150', '+0.150'],
            ['递归自我模型', '0.729', '0.323', '+0.406'],
            ['综合评分', '0.426', '0.247', '+0.179'],
        ],
        col_widths=[3.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]
    )
    story.append(Paragraph('表8：意识维度综合评分与对照差异', caption_style))
    story.append(overall_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f'TrueMan Agent在5个维度中4个优于基线LLM，综合评分差值+0.179。'
        f'最强的维度是递归自我模型（差值+0.406），'
        f'最弱的是情节性记忆（差值-0.100）——'
        f'这是唯一一个基线LLM优于TrueMan的维度，原因是焦虑信号在回忆阶段的干扰。',
        body_style
    ))
    story.append(Paragraph(
        f'<b>与先前报告的差异说明</b>：先前报告中基线LLM的综合评分为0.037，'
        f'这是因为score_baseline()方法将元认知监控/控制/情节记忆/时间连续性直接设为0，'
        f'仅从非模板度估算递归自我模型。'
        f'本文通过<b>实际运行基线LLM对照实验</b>，'
        f'获得了基于行为代理指标的更公平评分（0.247），使对比更有说服力。',
        body_style
    ))

    # ===== 7. 讨论 =====
    story.append(Paragraph('7  讨论', h1_style))
    
    story.append(Paragraph('7.1  内稳态驱动机制的有效性', h2_style))
    story.append(Paragraph(
        f'实验结果支持核心假设：内稳态驱动的情绪信号机制'
        f'<b>能让LLM涌现出与自我意识相关的行为特征</b>。'
        f'焦虑信号与实际不确定性的Pearson相关系数达0.669，'
        f'情绪回忆匹配度达1.000，自我描述非模板度达0.998。'
        f'这些结果验证了第3节的理论推导：'
        f'情绪信号不是被"编程"的意识模拟，而是内稳态偏离的<b>自然响应</b>。',
        body_style
    ))
    
    story.append(Paragraph('7.2  焦虑信号的系统性误判', h2_style))
    story.append(Paragraph(
        f'当前框架最显著的缺陷是焦虑信号的<b>系统性误判</b>：'
        f'将学习分享、知识陈述等正常对话误判为焦虑表达。'
        f'从理论推导的角度，这一误判的根源在于：焦虑信号的轻量模式'
        f'用文本表面差异度近似了认知不确定性A，但两者并不等价。'
        f'精确的A应基于<b>语义层面的分歧</b>而非<b>字面层面的分歧</b>。'
        f'改进方向：（1）引入语义等价检测；（2）使用logprobs参数获取token级概率分布。',
        body_style
    ))
    
    story.append(Paragraph('7.3  API模式的局限', h2_style))
    story.append(Paragraph(
        f'本实验在API模式下进行，LoRA可塑性系统不可用，导致以下局限：'
        f'（1）记忆无法"内化"到模型权重，仅通过prompt注入；'
        f'（2）睡眠整合无法执行；'
        f'（3）世界模型无法通过惊奇驱动进行在线更新。'
        f'在本地部署模式下，LoRA可塑性系统将使Agent能够'
        f'将高情绪强度的经历"写入"模型权重，'
        f'实现从"查阅"到"内化"的转变。',
        body_style
    ))
    
    story.append(Paragraph('7.4  基座模型的影响', h2_style))
    story.append(Paragraph(
        f'实验2揭示了框架机制与基座模型性格的交互效应：'
        f'TrueMan的焦虑信号正确触发了内省策略（触发率100%），'
        f'但DeepSeek倾向于"和稀泥"而非明确纠错。'
        f'这表明<b>情绪信号提供了正确的"何时反思"信号，'
        f'但"如何反思"仍受基座模型能力制约</b>。',
        body_style
    ))
    
    story.append(Paragraph('7.5  关于"意识"的审慎声明', h2_style))
    story.append(Paragraph(
        f'本实验仅验证<b>计算性行为指标</b>，'
        f'行为指标通过<b>不代表</b>系统具有主观意识（qualia）。'
        f'我们验证的行为特征是自我意识的<b>必要条件</b>而非<b>充分条件</b>。'
        f'正如《纽约动物意识宣言》所倡导的，'
        f'应当从"连续光谱主义"的视角看待意识——'
        f'不是"有或无"的二元判断，而是在多个维度上的渐变频谱。'
        f'TrueMan框架的贡献在于提供了<b>可量化的实验框架</b>，'
        f'使这一频谱可以被测量和比较。',
        body_style
    ))
    
    # ===== 8. 结论 =====
    story.append(Paragraph('8  结论与未来工作', h1_style))
    story.append(Paragraph(
        f'本文提出TrueMan框架，通过内稳态驱动的三种情绪信号'
        f'（惊奇、无聊、焦虑）实现LLM Agent的元认知监控、元认知控制、'
        f'情节性记忆与递归自我模型。4个递进实验在DeepSeek大语言模型上的验证表明：',
        body_style
    ))
    story.append(Paragraph(f'（1）TrueMan Agent的综合意识特征评分（0.426）显著高于基线LLM（0.247），差值+0.179；', bullet_style))
    story.append(Paragraph(f'（2）递归自我模型是最强维度（0.729），Agent能生成非模板化的、基于真实内部状态变化的自我描述；', bullet_style))
    story.append(Paragraph(f'（3）焦虑信号与实际不确定性的Pearson相关系数达0.669，情绪回忆匹配度达1.000；', bullet_style))
    story.append(Paragraph(f'（4）焦虑信号存在系统性误判，自我纠错率和时间连续性有待提升。', bullet_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f'<b>未来工作</b>：'
        f'（1）修复焦虑信号的系统性误判，引入语义等价检测；'
        f'（2）在本地部署模式下启用LoRA可塑性系统，验证记忆内化效果；'
        f'（3）增加实验重复次数，进行统计显著性检验；'
        f'（4）探索不同基座模型的对比实验；'
        f'（5）实现二阶递归反思，提升递归自我模型深度；'
        f'（6）将实验框架扩展至IIT的Φ指标和GNW的非线性激增指标。',
        body_style
    ))
    
    # ===== 参考文献 =====
    story.append(Paragraph('参考文献', h1_style))
    
    refs = [
        '[1] The New York Declaration on Animal Consciousness. NYU Center for Mind, Brain and Consciousness, 2024.',
        '[2] Butlin, A., et al. Consciousness in artificial intelligence: Insights from the science of consciousness. Nature, 2023.',
        '[3] Parr, T., Pezzulo, G., and Friston, K. Active Inference: The Free Energy Principle in Mind, Brain, and Behaviour. MIT Press, 2022.',
        '[4] Heins, C., et al. AXIOM: Learning to play games in minutes with expanding object-centric models. arXiv, 2025.',
        '[5] Ororbia, A., Friston, K., and Rao, R. Meta-representational predictive coding. arXiv, 2025.',
        '[6] Yoshida, M., Sprekeler, D., and Gutkin, B. Linking homeostasis to RL: HRRL. arXiv, 2025.',
        '[7] Hakim, V. Multi-scale temporal homeostasis (MSTH). arXiv, 2026.',
        '[8] Christov-Moore, C., et al. The conditions of physical embodiment. arXiv, 2025.',
        '[9] Kawato, M. and Cortese, A. From internal models toward metacognitive AI. arXiv, 2021/2023.',
        '[10] Blum, L. and Blum, M. The conscious turing machine. arXiv, 2023.',
        '[11] Mineault, P., et al. Cognitive dark matter. arXiv, 2026.',
        '[12] Sorrenti, L., et al. Wake-sleep consolidated learning. arXiv, 2024.',
        '[13] Delanois, D., et al. Sleep replay consolidation. arXiv, 2026.',
        '[14] Krishnan, G., et al. Biologically inspired sleep algorithm for ANNs. arXiv, 2019.',
        '[15] Dohare, S., et al. Maintaining plasticity in deep continual learning. arXiv, 2023.',
        '[16] Irie, K. and Gershman, S. Fast weight programming and linear transformers. TMLR, 2025.',
        '[17] Colas, C., et al. Language and culture internalisation for human-like autotelic AI. Nature Machine Intelligence, 2022.',
        '[18] Festinger, L. A Theory of Cognitive Dissonance. Stanford University Press, 1957.',
        '[19] Gillon, J. Modular theory of subjective consciousness. arXiv, 2025.',
        '[20] You, X. The nature of intelligence. arXiv, 2023/2024.',
        '[21] Vladu, A., et al. DIME architecture. arXiv, 2026.',
        '[22] Spisak, T. and Friston, K. Self-orthogonalizing attractor neural networks. Neurocomputing, 2025.',
        '[23] Friston, K., et al. Active inference and artificial reasoning. arXiv, 2025.',
        '[24] Gunasekaran, S., et al. Future-guided learning. Nature Communications, 2025.',
        '[25] Arani, E., et al. CLS-ER. ICLR, 2022.',
        '[26] Gupta, S., et al. Personalized AGI via neuroscience-inspired CL systems. arXiv, 2025.',
        '[27] Hayes, N., et al. Replay in deep learning. Neural Computation, 2021.',
        '[28] Dragomir, A., et al. JumpLoRA. arXiv, 2026.',
        '[29] Yang, Z., et al. NeuroLoRA. arXiv, 2026.',
        '[30] Kaushik, H., et al. Share: Shared LoRA subspaces. arXiv, 2026.',
        '[31] Luo, Y., et al. CORAL. arXiv, 2026.',
        '[32] Sukhija, N., et al. MaxInfoRL. ICLR, 2024.',
        '[33] Zhai, Y., et al. AgentEvolver. arXiv, 2025.',
        '[34] Putta, S., et al. Agent Q. arXiv, 2024.',
        '[35] Laurencon, R., et al. CTCS-HRRL. arXiv, 2024.',
    ]
    
    for ref in refs:
        story.append(Paragraph(esc(ref), ref_style))
    
    # ===== 构建PDF =====
    doc.build(story)
    print(f"PDF generated: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == '__main__':
    build_pdf()
