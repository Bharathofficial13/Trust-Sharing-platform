<?php
include("dbconnect.php");
session_start();
extract($_POST);
$cuid=$_SESSION['fid'];
?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Twitter Home</title>
	<link href="css/bootstrap.min.css" rel="stylesheet">
	<link href="css/font-awesome.min.css" rel="stylesheet">
	<link href="css/datepicker3.css" rel="stylesheet">
	<link href="css/styles.css" rel="stylesheet">
	
	<!--Custom Font-->
	<link href="https://fonts.googleapis.com/css?family=Montserrat:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">
	<!--[if lt IE 9]>
	<script src="js/html5shiv.js"></script>
	<script src="js/respond.min.js"></script>
	<![endif]-->
    <style type="text/css">
<!--
.im {border-radius: 50%;
}
.im1 {  border-radius: 50%;
}
.im2 {  border-radius: 50%;
}
.style1 {color: #000000}
-->
    </style>
</head>
<body>
	<nav class="navbar navbar-custom navbar-fixed-top" role="navigation">
		<div class="container-fluid">
			<div class="navbar-header">
				<button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#sidebar-collapse"><span class="sr-only">Toggle navigation</span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span></button>
				<a class="navbar-brand" href="#"><span>Twitter</span>Home</a>
				<ul class="nav navbar-top-links navbar-right">
				</ul>
			</div>
		</div><!-- /.container-fluid -->
	</nav>
	<div id="sidebar-collapse" class="col-sm-3 col-lg-2 sidebar">
		<div class="profile-sidebar">
			<div class="profile-userpic">
			<?php
					$qry=mysql_query("select * from register where id='$cuid'");
					$rw=mysql_fetch_array($qry); 
					?>
				<img src="img/<?php echo $rw['fname'];?>" class="img-responsive" alt="">
			</div>
			<div class="profile-usertitle">
				<div class="profile-usertitle-name"><?php echo $rw['name'];?></div>
				<div class="profile-usertitle-status"><span class="indicator label-success"></span>Online</div>
			</div>
			<div class="clear"></div>
		</div>
		<div class="divider"></div>
		<form role="search">
			<div class="form-group">
				<input type="text" class="form-control" placeholder="Search">
			</div>
		</form>
		<ul class="nav menu">
			<li  class="active"><a href="home.php"><em class="fa fa-dashboard">&nbsp;</em> Home</a></li>
			<li><a href="twit.php"><em class="fa fa-calendar">&nbsp;</em> Twit</a></li>
			<li><a href="follow.php"><em class="fa fa-bar-chart">&nbsp;</em> Follow List</a></li>
			<li><a href="ffriend.php"><em class="fa fa-toggle-off">&nbsp;</em> Find Friends</a></li>
			<li><a href="activity.php"><em class="fa fa-clone">&nbsp;</em> Notification</a></li>
			<li><a href="profile.php"><em class="fa fa-clone">&nbsp;</em> Profile</a></li>
			<li><a href="index.php"><em class="fa fa-power-off">&nbsp;</em> Logout</a></li>
		</ul>
	</div><!--/.sidebar-->
		
	<div class="col-sm-9 col-sm-offset-3 col-lg-10 col-lg-offset-2 main">
		<div class="row">
			<ol class="breadcrumb">
				<li><a href="#">
					<em class="fa fa-home"></em>
				</a></li>
				<li class="active">Forms</li>
			</ol>
		</div><!--/.row-->
		
		<div class="row">
			<div class="col-lg-12">
				<h1 class="page-header">Tweet List </h1>
			</div>
		</div><!--/.row-->
				
		
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading">List</div>
				  <div class="panel-body">
				    <form action="" method="post" name="form">
                      <?php
																	$qr=mysql_query("select * from frlist where frid='$cuid' && status='1'");
																	$ml=mysql_num_rows($qr);
																	if($ml>0)
																	{
																	while($row=mysql_fetch_array($qr))
																	{
																	$fiid=$row['id'];
																	$qt=mysql_query("select * from upost where sid='$fiid' or sid='$cuid' ORDER BY id DESC");
																	while($r=mysql_fetch_array($qt))
																	{
																	$rt=$r['sid'];
																	$g=mysql_query("select * from register where id='$rt'");
																	$ghh=mysql_fetch_array($g);
																	?>
                      <table width="91%" border="0">
                        <tr>
                          <td height="39">&nbsp;</td>
                          <td colspan="3"><img src="images/<?php echo $ghh['fname'];?>" width="25" height="25" class="im2">&nbsp;<?php echo $ghh['name'];?></td>
                          <td>&nbsp;</td>
                        </tr>
                        <tr>
                          <td height="32">&nbsp;</td>
                          <td colspan="3">Date:<?php echo $r['date'];?> Time:<?php echo $r['time'];?></td>
                          <td>&nbsp;</td>
                        </tr>
                        <tr>
                          <td height="32">&nbsp;</td>
                          <td colspan="3"><div align="left"><?php echo $r['cption'];?></div></td>
                          <td>&nbsp;</td>
                        </tr>
                        <tr>
                          <td width="14%" height="224">&nbsp;</td>
                          <td colspan="3" valign="top"><div align="left"><img src="image/<?php echo $r['image'];?>" width="250" height="200" ></div></td>
                          <td width="25%">&nbsp;</td>
                        </tr>
                        <tr>
                          <td>&nbsp;</td>
                          <td colspan="3">&nbsp;</td>
                          <td>&nbsp;</td>
                        </tr>
                        <tr>
                          <td>&nbsp;</td>
                          <td width="16%"><div align="center"><i class="fa fa-comments"></i><a href="comment.php?id=<?php echo $r['id'];?>">Reply</a></div></td>
                          <td width="15%"></td>
                          <td width="30%"><div align="left"><i class="fa fa-share-square"></i><a href="share.php?id=<?php echo $r['id'];?>">Retweet</a></div></td>
                          <td>&nbsp;</td>
                        </tr>
                      </table>
				      <?php
																	}
																	}
																	}
																	if($ml==0)
																	{
																	$qt1=mysql_query("select * from upost where sid='$cuid' ORDER BY id DESC");
																	while($r1=mysql_fetch_array($qt1))
																	{
																	$rt1=$r1['sid'];
																	$g1=mysql_query("select * from register where id='$rt1'");
																	$ghh1=mysql_fetch_array($g1);
																	?>
                      <table width="91%" border="0">
                        <tr>
                          <td height="39">&nbsp;</td>
                          <td colspan="3"><img src="images/<?php echo $ghh1['fname'];?>" width="25" height="25" class="im2">&nbsp;<?php echo $ghh1['name'];?></td>
                          <td>&nbsp;</td>
                        </tr>
                        <tr>
                          <td height="32">&nbsp;</td>
                          <td colspan="3">Date:<?php echo $r1['date'];?> Time:<?php echo $r1['time'];?></td>
                          <td>&nbsp;</td>
                        </tr>
                        <tr>
                          <td height="32">&nbsp;</td>
                          <td colspan="3"><div align="left"><?php echo $r1['cption'];?></div></td>
                          <td>&nbsp;</td>
                        </tr>
                        <tr>
                          <td width="16%" height="224">&nbsp;</td>
                          <td colspan="3" valign="top"><div align="left"><img src="image/<?php echo $r1['image'];?>" width="250" height="200" ></div></td>
                          <td width="30%">&nbsp;</td>
                        </tr>
                        <tr>
                          <td>&nbsp;</td>
                          <td colspan="3" class="style1">&nbsp;</td>
                          <td>&nbsp;</td>
                        </tr>
                        <tr>
                          <td>&nbsp;</td>
                          <td width="25%"><div align="center"><i class="fa fa-comments"></i><a href="comment.php?id=<?php echo $r1['id'];?>">Reply</a></div></td>
                          <td width="19%"></td>
                          <td width="10%"><div align="left"><i class="fa fa-share-square"></i><a href="share.php?id=<?php echo $r1['id'];?>">Retweet</a></div></td>
                          <td>&nbsp;</td>
                        </tr>
                      </table>
				      <?php
																	}
																	}
																	?>
                    </form>
				  </div>
				</div><!-- /.panel-->
			</div>
			<!-- /.panel-->
	  </div><!-- /.col-->
			<div class="col-sm-12">
				<p class="back-link">Lumino Theme by <a href="https://www.medialoot.com">Medialoot</a></p>
			</div>
</div><!-- /.row -->
	</div><!--/.main-->
	
<script src="js/jquery-1.11.1.min.js"></script>
	<script src="js/bootstrap.min.js"></script>
	<script src="js/chart.min.js"></script>
	<script src="js/chart-data.js"></script>
	<script src="js/easypiechart.js"></script>
	<script src="js/easypiechart-data.js"></script>
	<script src="js/bootstrap-datepicker.js"></script>
	<script src="js/custom.js"></script>
	
</body>
</html>
